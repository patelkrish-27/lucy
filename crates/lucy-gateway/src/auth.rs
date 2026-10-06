//! Pairing and device authentication.
//!
//! The gateway is reachable from the network, so it never trusts a connection
//! just because it bound to a private address. Two kinds of secret exist:
//!
//! - A **pairing token** minted by `lucy serve --pair` and shown once in a QR
//!   code. It is one-time: redeeming it converts it into a device token.
//! - A **device token** the app stores and presents on every reconnect.
//!
//! The desktop stores only a SHA-256 digest of either, so a stolen
//! `~/.config/lucy/gateway-devices.json` does not let an attacker in. The raw
//! token exists in exactly two places: the QR the user scans, and the phone's
//! secure storage.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Entropy in a token. 32 bytes is the same order as a session cookie and far
/// beyond guessing range at any request rate this server accepts.
const TOKEN_BYTES: usize = 32;

/// Prefixes make a leaked token recognizable in a log or paste without being
/// usable: the digest is what the server compares.
pub const PAIRING_PREFIX: &str = "lucy_pair_";
pub const DEVICE_PREFIX: &str = "lucy_dev_";

/// One paired phone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRecord {
    pub id: String,
    /// Human name for the phone, e.g. "Krish's Pixel".
    pub name: String,
    /// Hex SHA-256 of the device token.
    pub token_hash: String,
    pub created_at: u64,
    pub last_seen_at: u64,
}

impl DeviceRecord {
    pub fn verify(&self, token: &str) -> bool {
        // Constant-time is not strictly required for a hash comparison, but the
        // dependency-free XOR fold costs nothing and removes the argument.
        constant_time_eq(self.token_hash.as_bytes(), hash_token(token).as_bytes())
    }
}

/// Persisted state: known devices plus the current one-time pairing token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStore {
    /// Stable id for this desktop, so a phone that has seen more than one Lucy
    /// can tell them apart.
    pub server_id: String,
    /// Hash of the live pairing token, if one has been minted and not redeemed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_hash: Option<String>,
    /// Unix time the pairing token stops being redeemable. A QR left on a
    /// screen should not be a standing key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_expires_at: Option<u64>,
    /// Raw token kept only long enough to render the QR in the same process
    /// that minted it. Never written to disk.
    #[serde(skip)]
    pub pending_pairing_token: Option<String>,
    #[serde(default)]
    pub devices: BTreeMap<String, DeviceRecord>,
}

impl Default for DeviceStore {
    fn default() -> Self {
        Self {
            server_id: format!("srv-{}", random_hex(8)),
            pairing_hash: None,
            pairing_expires_at: None,
            pending_pairing_token: None,
            devices: BTreeMap::new(),
        }
    }
}

impl DeviceStore {
    /// Load from disk, or start a fresh store. A corrupt file is renamed aside
    /// rather than silently replaced: pairing again is safe, but losing the
    /// devices should not be invisible.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<DeviceStore>(&text) {
                Ok(store) => store,
                Err(e) => {
                    tracing::warn!(error=%e, path=%path.display(), "gateway device store unreadable; starting fresh");
                    rename_corrupt(path);
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            // The file holds token digests and a pairing digest. It should not
            // be world-readable on a shared machine.
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }

    pub fn is_paired(&self) -> bool {
        !self.devices.is_empty()
    }

    /// Mint a one-time pairing token, valid for `ttl_secs`. The returned string
    /// is shown in the QR and then forgotten; only its digest is persisted.
    pub fn mint_pairing(&mut self, ttl_secs: u64) -> String {
        let token = format!("{PAIRING_PREFIX}{}", random_hex(TOKEN_BYTES));
        self.pairing_hash = Some(hash_token(&token));
        self.pairing_expires_at = Some(now_secs().saturating_add(ttl_secs));
        self.pending_pairing_token = Some(token.clone());
        token
    }

    /// Force the pairing token to be considered expired.
    pub fn expire_pairing(&mut self) {
        self.pairing_hash = None;
        self.pairing_expires_at = None;
        self.pending_pairing_token = None;
    }

    /// True when a pairing token exists and has not expired.
    pub fn pairing_live(&self) -> bool {
        self.pairing_hash.is_some()
            && self
                .pairing_expires_at
                .map(|t| now_secs() < t)
                .unwrap_or(true)
    }

    /// Redeem a pairing token. On success the caller receives the new device
    /// token *once*; the store keeps only its digest. An expired token is
    /// refused and cleared.
    pub fn redeem_pairing(&mut self, presented: &str, name: &str) -> Option<(DeviceRecord, String)> {
        if !self.pairing_live() {
            self.expire_pairing();
            return None;
        }
        let expected = self.pairing_hash.as_deref()?;
        if !constant_time_eq(expected.as_bytes(), hash_token(presented).as_bytes()) {
            return None;
        }
        // One-time: a redeemed token can never be replayed.
        self.expire_pairing();
        Some(self.register_device(name))
    }

    /// Create a device token without a pairing round-trip. Used by tests and by
    /// an explicit `lucy serve --pair --reissue` path.
    pub fn register_device(&mut self, name: &str) -> (DeviceRecord, String) {
        let token = format!("{DEVICE_PREFIX}{}", random_hex(TOKEN_BYTES));
        let now = now_secs();
        let record = DeviceRecord {
            id: format!("dev-{}", random_hex(6)),
            name: if name.trim().is_empty() {
                "phone".to_string()
            } else {
                name.trim().to_string()
            },
            token_hash: hash_token(&token),
            created_at: now,
            last_seen_at: now,
        };
        self.devices.insert(record.id.clone(), record.clone());
        (record, token)
    }

    /// Resolve a presented device token to a record, if it is known.
    pub fn authenticate(&self, presented: &str) -> Option<DeviceRecord> {
        let digest = hash_token(presented);
        self.devices
            .values()
            .find(|d| constant_time_eq(d.token_hash.as_bytes(), digest.as_bytes()))
            .cloned()
    }

    /// Revoke one device by id. Returns true when a device was removed.
    pub fn revoke(&mut self, id: &str) -> bool {
        self.devices.remove(id).is_some()
    }

    pub fn touch(&mut self, id: &str) {
        if let Some(d) = self.devices.get_mut(id) {
            d.last_seen_at = now_secs();
        }
    }
}

/// SHA-256 hex digest of a token.
pub fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut buf);
    hex::encode(buf)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn rename_corrupt(path: &Path) {
    let mut bak = path.as_os_str().to_owned();
    bak.push(format!(".corrupt-{}.bak", now_secs()));
    let _ = std::fs::rename(path, PathBuf::from(bak));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_store_is_unpaired_and_has_a_stable_id() {
        let store = DeviceStore::default();
        assert!(!store.is_paired());
        assert!(store.server_id.starts_with("srv-"));
        assert_eq!(store.server_id.len(), "srv-".len() + 16);
    }

    #[test]
    fn pairing_redeems_exactly_once() {
        let mut store = DeviceStore::default();
        let token = store.mint_pairing(600);
        assert!(token.starts_with(PAIRING_PREFIX));

        let (record, device_token) = store
            .redeem_pairing(&token, "Krish's Pixel")
            .expect("first redemption succeeds");
        assert!(device_token.starts_with(DEVICE_PREFIX));
        assert_eq!(record.name, "Krish's Pixel");
        assert!(store.is_paired());

        // A second redemption of the same token fails: it is one-time.
        assert!(store.redeem_pairing(&token, "attacker").is_none());
        // And reassigning the pending raw token does not resurrect it.
        assert!(store.redeem_pairing("", "attacker").is_none());
    }

    #[test]
    fn an_expired_pairing_token_is_refused() {
        let mut store = DeviceStore::default();
        let token = store.mint_pairing(600);
        // Pretend the TTL passed.
        store.pairing_expires_at = Some(now_secs().saturating_sub(1));
        assert!(store.redeem_pairing(&token, "phone").is_none());
        assert!(!store.pairing_live());
        assert!(store.pairing_hash.is_none(), "expiry clears the digest");
    }

    #[test]
    fn a_wrong_pairing_token_never_redeems() {
        let mut store = DeviceStore::default();
        store.mint_pairing(600);
        assert!(store.redeem_pairing("lucy_pair_wrong", "attacker").is_none());
        assert!(!store.is_paired());
        // The real token is still live, so a typo does not force a re-pair.
        assert!(store.pairing_live());
    }

    #[test]
    fn a_device_token_authenticates_and_a_wrong_one_does_not() {
        let mut store = DeviceStore::default();
        let (record, token) = store.register_device("phone");
        assert_eq!(store.authenticate(&token).map(|d| d.id), Some(record.id.clone()));
        assert!(store.authenticate("lucy_dev_nope").is_none());
        // A pairing token is not a device token.
        let pairing = store.mint_pairing(600);
        assert!(store.authenticate(&pairing).is_none());
    }

    #[test]
    fn the_raw_token_is_never_stored() {
        let mut store = DeviceStore::default();
        let (record, token) = store.register_device("phone");
        assert_eq!(record.token_hash, hash_token(&token));
        assert_ne!(record.token_hash, token, "only the digest may persist");

        let minted = store.mint_pairing(600);
        assert_ne!(store.pairing_hash.as_deref(), Some(minted.as_str()));
        assert_eq!(store.pairing_hash.as_deref(), Some(hash_token(&minted).as_str()));
    }

    #[test]
    fn revoke_removes_only_the_named_device() {
        let mut store = DeviceStore::default();
        let (a, _) = store.register_device("a");
        let (b, _) = store.register_device("b");
        assert!(store.revoke(&a.id));
        assert!(!store.revoke(&a.id), "double revoke is a no-op");
        assert_eq!(store.devices.len(), 1);
        assert!(store.devices.contains_key(&b.id));
    }

    #[test]
    fn store_round_trips_without_the_pending_raw_token() {
        let dir = std::env::temp_dir().join(format!("lucy-gw-auth-{}", now_secs()));
        let path = dir.join("devices.json");
        let mut store = DeviceStore::default();
        let minted = store.mint_pairing(600);
        let (record, token) = store.register_device("phone");
        store.save(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains(&minted) && !text.contains(&token),
            "no raw secret may reach disk"
        );

        let back = DeviceStore::load(&path);
        assert_eq!(back.server_id, store.server_id);
        assert!(back.authenticate(&token).is_some());
        assert_eq!(back.devices[&record.id].name, "phone");
        // The one-time token is still redeemable after a restart, but only
        // because its digest persisted — the raw form did not.
        assert!(back.pairing_hash.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_file_is_renamed_and_replaced() {
        let dir = std::env::temp_dir().join(format!("lucy-gw-corrupt-{}", now_secs()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("devices.json");
        std::fs::write(&path, "{not json").unwrap();
        let store = DeviceStore::load(&path);
        assert!(!store.is_paired());
        assert!(!path.exists(), "corrupt file moved aside");
        let backups: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".corrupt-"))
            .collect();
        assert_eq!(backups.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
