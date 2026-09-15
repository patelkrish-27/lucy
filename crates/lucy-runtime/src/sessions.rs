//! Session management backed exclusively by ADK SessionService.
use anyhow::Result;
use lucy_adk::LucySessionService;
use lucy_core::{SessionId, SessionMeta, TurnMessage};
use super::LucyRuntime;

pub(crate) fn now()->u64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()}

/// Keep the newest turns in the runtime cache while preserving the initial task anchor.
pub fn trim_history(history:&mut Vec<TurnMessage>,max_history:usize){if history.len()<=max_history{return;}let drop_n=history.len()-max_history;if history.first().map_or(false,|m|matches!(m,TurnMessage::User(_)))&&history.len()>1{let drain_count=drop_n.min(history.len()-1);history.drain(1..1+drain_count);}else{history.drain(0..drop_n);}}

impl LucyRuntime {
    pub fn sessions_dir(&self)->&std::path::Path{self.session_service.db_path()}
    pub async fn current_meta(&self)->SessionMeta{SessionMeta::from(&*self.session.lock().await)}
    pub async fn current_id(&self)->SessionId{self.session.lock().await.session_id.clone()}
    pub async fn list_sessions(&self)->Result<Vec<SessionMeta>>{self.session_service.list().await}
    pub async fn new_session(&self,title:Option<String>)->Result<SessionMeta>{self.interrupt.reset();let s=self.session_service.create(title).await?;let meta=SessionMeta::from(&s);*self.session.lock().await=s;Ok(meta)}
    pub async fn switch_session(&self,id:&SessionId)->Result<SessionMeta>{self.interrupt.reset();let s=self.session_service.load(id).await?;let meta=SessionMeta::from(&s);*self.session.lock().await=s;Ok(meta)}
    pub async fn rename_current(&self,title:String)->Result<SessionMeta>{let id=self.session.lock().await.session_id.clone();self.session_service.update_title(&id,title).await}
    pub async fn delete_session(&self,id:&SessionId)->Result<bool>{let current=self.session.lock().await.session_id.clone();let was_current=current==*id;self.session_service.delete(id).await?;if was_current{let remaining=self.session_service.list().await?;let next=if let Some(m)=remaining.first(){self.session_service.load(&m.id).await?}else{self.session_service.create(None).await?};*self.session.lock().await=next;}Ok(was_current)}
    pub async fn fork_current(&self)->Result<SessionMeta>{let src=self.session.lock().await.clone();let forked=self.session_service.fork(&src).await?;let meta=SessionMeta::from(&forked);*self.session.lock().await=forked;Ok(meta)}
    pub async fn clear_session(&self)->Result<()>{let current=self.session.lock().await.clone();let fresh=self.session_service.clear(&current.session_id,current.title.clone()).await?;*self.session.lock().await=fresh;Ok(())}
    pub async fn clear_current(&self)->Result<SessionMeta>{self.clear_session().await?;Ok(self.current_meta().await)}
    pub async fn compact(&self)->Result<String>{let max=self.config.sessions.max_history;let mut s=self.session.lock().await;let before=s.history.len();if before<=max{return Ok("nothing to compact".to_string());}trim_history(&mut s.history,max);Ok(format!("compacted {before} -> {} messages in runtime context; ADK event history remains append-only",s.history.len()))}
    pub async fn compact_current(&self,keep:usize)->Result<(usize,usize)>{let mut s=self.session.lock().await;let before=s.history.len();trim_history(&mut s.history,keep.max(1));Ok((before,s.history.len()))}
    pub async fn export_current(&self,path:std::path::PathBuf)->Result<()>{let s=self.session_service.load(&self.session.lock().await.session_id).await?;s.save_to_file(&path).await}
    pub async fn session_service(&self)->LucySessionService{(*self.session_service).clone()}
}

#[cfg(test)]
mod tests{use super::*;#[test]fn trim_history_preserves_initial_user_prompt(){let mut hist=vec![TurnMessage::User("initial task".into()),TurnMessage::User("intermediate 1".into()),TurnMessage::User("intermediate 2".into()),TurnMessage::User("intermediate 3".into()),TurnMessage::User("latest turn".into())];trim_history(&mut hist,3);assert_eq!(hist.len(),3);assert!(matches!(&hist[0],TurnMessage::User(t)if t=="initial task"));assert!(matches!(&hist[1],TurnMessage::User(t)if t=="intermediate 3"));assert!(matches!(&hist[2],TurnMessage::User(t)if t=="latest turn"));}}
