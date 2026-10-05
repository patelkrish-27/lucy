package com.patelkrish.lucy.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

private val LucyBg = Color(0xFF101116)
private val LucySurface = Color(0xFF181A20)
private val LucySurface2 = Color(0xFF20232B)
private val LucyAccent = Color(0xFF9B7BFF)
private val LucyText = Color(0xFFF3F1F7)
private val LucyMuted = Color(0xFFA5A1AF)

private data class PoseRect(val x: Int, val y: Int, val width: Int, val height: Int)

private val PoseRects = mapOf(
        "angry" to PoseRect(0, 0, 86, 86),
        "blink" to PoseRect(109, 0, 89, 95),
        "bored" to PoseRect(219, 0, 86, 84),
        "breathing" to PoseRect(328, 0, 89, 95),
        "celebrate" to PoseRect(438, 0, 104, 102),
        "cheer" to PoseRect(547, 0, 104, 103),
        "chill" to PoseRect(657, 0, 98, 95),
        "clap" to PoseRect(0, 112, 104, 103),
        "click" to PoseRect(109, 112, 73, 88),
        "confused" to PoseRect(219, 112, 89, 97),
        "crying" to PoseRect(328, 112, 86, 87),
        "dance" to PoseRect(438, 112, 104, 103),
        "debugging" to PoseRect(547, 112, 104, 103),
        "disappointed" to PoseRect(657, 112, 86, 84),
        "drag" to PoseRect(0, 223, 73, 88),
        "embarrassed" to PoseRect(109, 223, 86, 84),
        "error" to PoseRect(219, 223, 73, 88),
        "excited_wiggle" to PoseRect(328, 223, 88, 95),
        "explaining" to PoseRect(438, 223, 104, 105),
        "explaining_talking" to PoseRect(547, 223, 89, 99),
        "focused" to PoseRect(657, 223, 104, 105),
        "frustrated" to PoseRect(0, 335, 86, 86),
        "happy" to PoseRect(109, 335, 104, 102),
        "heart" to PoseRect(219, 335, 104, 102),
        "hide" to PoseRect(328, 335, 73, 86),
        "hover" to PoseRect(438, 335, 89, 95),
        "hurt" to PoseRect(547, 335, 86, 86),
        "idea" to PoseRect(657, 335, 104, 106),
        "idle" to PoseRect(0, 446, 88, 95),
        "laugh" to PoseRect(109, 446, 104, 102),
        "lie_down" to PoseRect(219, 446, 89, 95),
        "listening" to PoseRect(328, 446, 98, 94),
        "loading" to PoseRect(438, 446, 73, 88),
        "lonely" to PoseRect(547, 446, 86, 86),
        "look_around" to PoseRect(657, 446, 89, 95),
        "love" to PoseRect(0, 558, 104, 104),
        "nodding" to PoseRect(109, 558, 98, 93),
        "notification" to PoseRect(219, 558, 73, 86),
        "overwhelmed" to PoseRect(328, 558, 86, 86),
        "peek" to PoseRect(438, 558, 73, 86),
        "planning" to PoseRect(547, 558, 104, 89),
        "pop_in" to PoseRect(657, 558, 73, 88),
        "pop_out" to PoseRect(0, 670, 73, 86),
        "processing" to PoseRect(109, 670, 73, 88),
        "progress" to PoseRect(219, 670, 104, 105),
        "question" to PoseRect(328, 670, 88, 97),
        "reading" to PoseRect(438, 670, 104, 104),
        "sad" to PoseRect(547, 670, 86, 87),
        "scared" to PoseRect(657, 670, 86, 84),
        "search" to PoseRect(0, 781, 73, 87),
        "shy" to PoseRect(109, 781, 88, 99),
        "singing" to PoseRect(219, 781, 99, 95),
        "sit" to PoseRect(328, 781, 89, 95),
        "sparkle" to PoseRect(438, 781, 104, 102),
        "spin" to PoseRect(547, 781, 89, 95),
        "success" to PoseRect(657, 781, 104, 104),
        "success_tick" to PoseRect(0, 893, 73, 86),
        "surprised" to PoseRect(109, 893, 88, 99),
        "talking" to PoseRect(219, 893, 88, 99),
        "tap" to PoseRect(328, 893, 73, 88),
        "thinking_code" to PoseRect(438, 893, 104, 103),
        "thumbs_up" to PoseRect(547, 893, 104, 102),
        "tired" to PoseRect(657, 893, 86, 87),
        "typing" to PoseRect(0, 1004, 104, 105),
        "vibing" to PoseRect(109, 1004, 99, 96),
        "wave" to PoseRect(219, 1004, 89, 95),
        "zoom_in" to PoseRect(328, 1004, 73, 88)
)

private fun loadPoseAtlas(context: Context): Bitmap? =
    runCatching {
        context.assets.open("lucy_pose_atlas.webp").use { BitmapFactory.decodeStream(it) }
    }.getOrNull()

@Composable
private fun LucyMascot(
    pose: String,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val atlas = remember(context) { loadPoseAtlas(context) }
    val rect = PoseRects[pose] ?: PoseRects.getValue("idle")

    if (atlas == null) {
        Box(modifier, contentAlignment = Alignment.Center) {
            Text("L", color = LucyAccent, fontSize = 48.sp, fontWeight = FontWeight.Black)
        }
        return
    }

    val cropped = remember(atlas, pose) {
        val safeX = rect.x.coerceIn(0, atlas.width - 1)
        val safeY = rect.y.coerceIn(0, atlas.height - 1)
        val safeW = rect.width.coerceAtMost(atlas.width - safeX)
        val safeH = rect.height.coerceAtMost(atlas.height - safeY)
        Bitmap.createBitmap(atlas, safeX, safeY, safeW, safeH)
    }

    Image(
        bitmap = cropped.asImageBitmap(),
        contentDescription = "Lucy $pose pose",
        contentScale = ContentScale.Fit,
        modifier = modifier.then(if (onClick != null) Modifier.clickable { onClick() } else Modifier)
    )
}

@Composable
fun LucyApp() {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    var screen by remember { mutableStateOf("Chat") }

    MaterialTheme(
        colorScheme = MaterialTheme.colorScheme.copy(
            background = LucyBg,
            surface = LucySurface,
            surfaceVariant = LucySurface2,
            primary = LucyAccent,
            onBackground = LucyText,
            onSurface = LucyText,
            onSurfaceVariant = LucyMuted,
        )
    ) {
        ModalNavigationDrawer(
            drawerState = drawerState,
            drawerContent = {
                ModalDrawerSheet(
                    drawerContainerColor = LucySurface,
                    windowInsets = WindowInsets.safeDrawing,
                ) {
                    DrawerHeader()
                    Spacer(Modifier.padding(8.dp))
                    NavigationDrawerItem(label = { Text("Chat") }, selected = screen == "Chat",
                        onClick = { screen = "Chat"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Chat, null) })
                    NavigationDrawerItem(label = { Text("Sessions") }, selected = screen == "Sessions",
                        onClick = { screen = "Sessions"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.History, null) })
                    NavigationDrawerItem(label = { Text("Models") }, selected = screen == "Models",
                        onClick = { screen = "Models"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Psychology, null) })
                    NavigationDrawerItem(label = { Text("Tools") }, selected = screen == "Tools",
                        onClick = { screen = "Tools"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Build, null) })
                    NavigationDrawerItem(label = { Text("Files") }, selected = screen == "Files",
                        onClick = { screen = "Files"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Folder, null) })
                    NavigationDrawerItem(label = { Text("Settings") }, selected = screen == "Settings",
                        onClick = { screen = "Settings"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Settings, null) })
                }
            }
        ) {
            when (screen) {
                "Chat" -> ChatScreen { scope.launch { drawerState.open() } }
                else -> PlaceholderScreen(screen) { screen = "Chat" }
            }
        }
    }
}

@Composable
private fun DrawerHeader() {
    Column(Modifier.fillMaxWidth().padding(20.dp)) {
        Text("LUCY", color = LucyText, fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.padding(3.dp))
        Text("Your AI Computer Buddy", color = LucyMuted, fontSize = 13.sp)
    }
}

private data class UiMessage(val role: String, val text: String)

private val poseCycle = listOf(
    "idle", "thinking_code", "typing", "processing", "search", "explaining",
    "happy", "celebrate", "heart", "wave", "dance", "singing", "sad", "confused"
)

@Composable
private fun ChatScreen(onOpenDrawer: () -> Unit) {
    val messages = remember {
        mutableStateListOf(UiMessage("assistant", "Hey, I’m Lucy. What should we get done?"))
    }
    var input by remember { mutableStateOf("") }
    var pose by remember { mutableStateOf("idle") }

    Surface(Modifier.fillMaxSize(), color = LucyBg) {
        Column(Modifier.fillMaxSize().imePadding()) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onOpenDrawer) {
                    Icon(Icons.Default.Menu, "Open navigation", tint = LucyText)
                }
                Column(Modifier.weight(1f)) {
                    Text("Lucy", color = LucyText, fontWeight = FontWeight.SemiBold)
                    Text("Ready", color = Color(0xFF71D88B), fontSize = 12.sp)
                }
                IconButton(onClick = { pose = "thinking_code" }) {
                    Icon(Icons.Default.MoreVert, "More", tint = LucyText)
                }
            }

            Box(
                Modifier.fillMaxWidth().padding(top = 2.dp, bottom = 2.dp),
                contentAlignment = Alignment.Center
            ) {
                LucyMascot(
                    pose = pose,
                    modifier = Modifier.size(150.dp),
                    onClick = {
                        val next = poseCycle[(poseCycle.indexOf(pose).coerceAtLeast(0) + 1) % poseCycle.size]
                        pose = next
                    }
                )
            }

            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(messages) { message -> MessageBubble(message) }
            }

            Composer(
                value = input,
                onValueChange = {
                    input = it
                    pose = if (it.isBlank()) "idle" else "typing"
                },
                onSend = {
                    val text = input.trim()
                    if (text.isNotEmpty()) {
                        messages += UiMessage("user", text)
                        pose = "processing"
                        messages += UiMessage("assistant", "I’ve got it. I’ll work through that task.")
                        input = ""
                        pose = "success"
                    }
                },
                onMic = { pose = "listening" }
            )
        }
    }
}

@Composable
private fun MessageBubble(message: UiMessage) {
    val user = message.role == "user"
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (user) Arrangement.End else Arrangement.Start
    ) {
        Surface(
            color = if (user) LucyAccent else LucySurface,
            shape = RoundedCornerShape(16.dp)
        ) {
            Text(
                message.text,
                color = if (user) Color(0xFF120F18) else LucyText,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                fontSize = 16.sp
            )
        }
    }
}

@Composable
private fun Composer(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    onMic: () -> Unit,
) {
    Surface(
        Modifier.fillMaxWidth().padding(12.dp),
        color = LucySurface,
        shape = RoundedCornerShape(20.dp)
    ) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = {}) {
                Icon(Icons.Default.Add, "Attach", tint = LucyMuted)
            }
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.weight(1f).padding(horizontal = 6.dp, vertical = 10.dp),
                textStyle = MaterialTheme.typography.bodyLarge.copy(color = LucyText),
                decorationBox = { inner ->
                    if (value.isEmpty()) Text("Message Lucy…", color = LucyMuted)
                    inner()
                }
            )
            IconButton(onClick = onMic) {
                Icon(Icons.Default.Mic, "Voice", tint = LucyMuted)
            }
            IconButton(onClick = onSend, enabled = value.isNotBlank()) {
                Icon(Icons.Default.Send, "Send", tint = if (value.isNotBlank()) LucyAccent else LucyMuted)
            }
        }
    }
}

@Composable
private fun PlaceholderScreen(title: String, onBack: () -> Unit) {
    Column(
        Modifier.fillMaxSize().background(LucyBg).padding(WindowInsets.safeDrawing.asPaddingValues())
    ) {
        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = LucyText)
            }
            Text(title, color = LucyText, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
        }
        Column(
            Modifier.fillMaxSize().padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("Lucy $title", color = LucyText, fontSize = 24.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.padding(6.dp))
            Text("Mobile surface coming next.", color = LucyMuted)
        }
    }
}
