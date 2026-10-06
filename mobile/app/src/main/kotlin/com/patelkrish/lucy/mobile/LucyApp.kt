package com.patelkrish.lucy.mobile

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
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Code
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

private val LucyBg = Color(0xFF101116)
private val LucySurface = Color(0xFF181A20)
private val LucySurface2 = Color(0xFF20232B)
private val LucyAccent = Color(0xFF9B7BFF)
private val LucyText = Color(0xFFF3F1F7)
private val LucyMuted = Color(0xFFA5A1AF)

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
                    NavigationDrawerItem(
                        label = { Text("Chat") },
                        selected = screen == "Chat",
                        onClick = { screen = "Chat"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Chat, null) }
                    )
                    NavigationDrawerItem(
                        label = { Text("Sessions") },
                        selected = screen == "Sessions",
                        onClick = { screen = "Sessions"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.History, null) }
                    )
                    NavigationDrawerItem(
                        label = { Text("Models") },
                        selected = screen == "Models",
                        onClick = { screen = "Models"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Psychology, null) }
                    )
                    NavigationDrawerItem(
                        label = { Text("Tools") },
                        selected = screen == "Tools",
                        onClick = { screen = "Tools"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Build, null) }
                    )
                    NavigationDrawerItem(
                        label = { Text("Files") },
                        selected = screen == "Files",
                        onClick = { screen = "Files"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Folder, null) }
                    )
                    NavigationDrawerItem(
                        label = { Text("Settings") },
                        selected = screen == "Settings",
                        onClick = { screen = "Settings"; scope.launch { drawerState.close() } },
                        icon = { Icon(Icons.Default.Settings, null) }
                    )
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

@Composable
private fun ChatScreen(onOpenDrawer: () -> Unit) {
    val messages = remember {
        mutableStateListOf(
            UiMessage("assistant", "Hey, I’m Lucy. What should we get done?")
        )
    }
    var input by remember { mutableStateOf("") }

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
                IconButton(onClick = {}) {
                    Icon(Icons.Default.MoreVert, "More", tint = LucyText)
                }
            }

            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                reverseLayout = false
            ) {
                items(messages) { message ->
                    MessageBubble(message)
                }
            }

            Composer(
                value = input,
                onValueChange = { input = it },
                onSend = {
                    val text = input.trim()
                    if (text.isNotEmpty()) {
                        messages += UiMessage("user", text)
                        messages += UiMessage("assistant", "I’ve got it. I’ll work through that task.")
                        input = ""
                    }
                }
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
) {
    Surface(
        Modifier.fillMaxWidth().padding(12.dp),
        color = LucySurface,
        shape = RoundedCornerShape(20.dp)
    ) {
        Column(Modifier.padding(top = 8.dp, bottom = 8.dp)) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 14.dp),
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
                IconButton(onClick = {}) {
                    Icon(Icons.Default.Mic, "Voice", tint = LucyMuted)
                }
                IconButton(onClick = onSend, enabled = value.isNotBlank()) {
                    Icon(Icons.Default.Send, "Send", tint = if (value.isNotBlank()) LucyAccent else LucyMuted)
                }
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
