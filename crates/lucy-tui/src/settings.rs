use lucy_config::LucyConfig;
use ratatui::{layout::{Alignment,Constraint,Direction,Layout,Rect},style::{Color,Modifier,Style},text::{Line,Span},widgets::{Block,Borders,Clear,List,ListItem,Paragraph},Frame};

pub const ITEMS: [&str; 16] = [
    "Main model (OpenChat)","Cheap model (Gemini 3.5 Flash-Lite)","Main API Key — OpenChat","Main Endpoint — OpenChat",
    "Cheap API Key — Gemini","Cheap Endpoint — Gemini","Planner max subtasks","Planner max depth",
    "HyprFast candidates","HyprFast batching","HyprFast parallel","Verify actions",
    "Voice model","Voice API Key","Voice hotkey","Session resume"
];

fn mask(k:&Option<String>)->String{
    match k{
        None=>"(not set)".into(),
        Some(s) if s.trim().is_empty()=>"(not set)".into(),
        Some(s) if s.len()<=8=>"***".into(),
        Some(s)=>format!("{}...{} ({} chars)", &s[..4], &s[s.len()-4..], s.len()),
    }
}

pub fn draw(frame:&mut Frame<'_>, area:Rect, cfg:&LucyConfig, selected:usize) {
    let popup=Rect{x:area.width/10,y:area.height/10,width:area.width*8/10,height:area.height*8/10};
    let popup_style = Style::default().bg(Color::Rgb(25, 25, 35)).fg(Color::White);
    let block_style = Style::default().bg(Color::Rgb(25, 25, 35)).fg(Color::White);
    let block=Block::default()
        .title(" Lucy Settings — any LLM brand via API Key + Endpoint ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan).bg(Color::Rgb(25, 25, 35)))
        .style(popup_style);
    frame.render_widget(Clear, popup);
    frame.render_widget(block.clone(),popup);
    let inner=block.inner(popup);
    frame.render_widget(Block::default().style(popup_style), inner);
    let cols=Layout::default().direction(Direction::Horizontal).constraints([Constraint::Percentage(45),Constraint::Percentage(55)]).split(inner);
    let items=ITEMS.iter().enumerate().map(|(i,s)|{
        let style = if i==selected {
            Style::default().bg(Color::Rgb(45, 45, 65)).fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().bg(Color::Rgb(25, 25, 35)).fg(Color::White)
        };
        ListItem::new(Line::from(vec![if i==selected{Span::styled("▶ ", style)}else{Span::styled("  ", style)},Span::styled(*s, style)])).style(style)
    }).collect::<Vec<_>>();
    let list_block = Block::default().borders(Borders::ALL).border_style(block_style).title(" Option ").style(popup_style);
    let list_area = cols[0];
    frame.render_widget(List::new(items).block(list_block).style(popup_style), list_area);

    let is_text_field = matches!(selected, 0|1|2|3|4|5|12|13|14);
    let raw = raw_value(cfg, selected);
    let display_value = match selected {
        2|4|13 => {
            if raw.is_empty() { "(not set) — type to edit".into() } else { raw.clone() }
        },
        3|5 => {
            if raw.is_empty() {
                if selected==3 { "https://api.openchat.ai/v1 (default) — type endpoint".into() }
                else { "https://generativelanguage.googleapis.com/v1beta/openai/ (Gemini)".into() }
            } else { raw.clone() }
        },
        _ if is_text_field => {
            if raw.is_empty() { "(empty) — type to edit".into() } else { raw.clone() }
        },
        _ => {
            let all_values=[
                cfg.models.main.clone(),
                cfg.models.hyprfast_command.clone(),
                mask(&cfg.models.main_api_key.clone().or_else(|| cfg.models.api_key.clone())),
                cfg.models.main_base_url.clone().or_else(|| cfg.models.base_url.clone()).unwrap_or_else(||"https://api.openchat.ai/v1".into()),
                mask(&cfg.models.cheap_api_key.clone().or_else(|| cfg.models.api_key.clone())),
                cfg.models.cheap_base_url.clone().or_else(|| cfg.models.base_url.clone()).unwrap_or_else(||"https://generativelanguage.googleapis.com/v1beta/openai/".into()),
                cfg.planner.max_subtasks.to_string(),
                cfg.planner.max_depth.to_string(),
                cfg.hyprfast.max_candidates.to_string(),
                cfg.hyprfast.batching.to_string(),
                cfg.hyprfast.parallel.to_string(),
                cfg.hyprfast.verify_actions.to_string(),
                cfg.voice.model.clone(),
                mask(&cfg.voice.api_key),
                cfg.voice.push_to_talk.clone(),
                cfg.sessions.resume.to_string()
            ];
            all_values[selected].clone()
        }
    };
    let value_title = if is_text_field { " Value (typing edits immediately) " } else { " Value " };
    let detail_block = Block::default().title(value_title).borders(Borders::ALL).border_style(block_style).style(popup_style);
    let detail_area = cols[1];
    frame.render_widget(detail_block.clone(), detail_area);
    let inner_detail = detail_block.inner(detail_area);
    let value_paragraph = Paragraph::new(display_value.clone())
        .alignment(Alignment::Center)
        .style(Style::default().bg(Color::Rgb(30, 30, 45)).fg(if is_text_field { Color::Yellow } else { Color::White }))
        .wrap(ratatui::widgets::Wrap{trim:false});
    frame.render_widget(value_paragraph, inner_detail);
    if is_text_field {
        let cursor_x = inner_detail.x + (inner_detail.width.saturating_sub(display_value.len() as u16 + 2)) / 2 + display_value.len() as u16 + 1;
        let cursor_y = inner_detail.y + inner_detail.height / 2;
        frame.set_cursor_position((cursor_x.min(inner_detail.right().saturating_sub(1)), cursor_y));
    }
    let help=Paragraph::new("↑/↓ select  ←/→ change  type to edit  Backspace deletes  Esc close  Ctrl+S save")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Gray).bg(Color::Rgb(25, 25, 35)).add_modifier(Modifier::DIM));
    let help_area=Rect{x:popup.x+1,y:popup.bottom().saturating_sub(2),width:popup.width.saturating_sub(2),height:1};
    frame.render_widget(help,help_area);
}

pub fn change(cfg:&mut LucyConfig, selected:usize, direction:i8) {
    match selected {
        6=>{let v=cfg.planner.max_subtasks as i32+direction as i32;cfg.planner.max_subtasks=v.clamp(1,256) as usize;}
        7=>{let v=cfg.planner.max_depth as i32+direction as i32;cfg.planner.max_depth=v.clamp(1,64) as usize;}
        8=>{let v=cfg.hyprfast.max_candidates as i32+direction as i32;cfg.hyprfast.max_candidates=v.clamp(1,68) as usize;}
        9=>cfg.hyprfast.batching=!cfg.hyprfast.batching,
        10=>cfg.hyprfast.parallel=!cfg.hyprfast.parallel,
        11=>cfg.hyprfast.verify_actions=!cfg.hyprfast.verify_actions,
        15=>cfg.sessions.resume=!cfg.sessions.resume,
        _=>{}
    }
}
pub fn edit_text(cfg:&mut LucyConfig,selected:usize,value:&str){
    let v=value.trim().to_owned();
    match selected{
        0=>cfg.models.main=v,
        1=>cfg.models.hyprfast_command=v,
        2=>cfg.models.main_api_key=if v.is_empty()||v=="(not set)"{None}else{Some(v)},
        3=>cfg.models.main_base_url=if v.is_empty()||v=="(not set)"||v.contains("(default)"){None}else{Some(v)},
        4=>cfg.models.cheap_api_key=if v.is_empty()||v=="(not set)"{None}else{Some(v)},
        5=>cfg.models.cheap_base_url=if v.is_empty()||v=="(not set)"||v.contains("(default)"){None}else{Some(v)},
        12=>cfg.voice.model=v,
        13=>cfg.voice.api_key=if v.is_empty()||v=="(not set)"{None}else{Some(v)},
        14=>cfg.voice.push_to_talk=v,
        _=>{}
    }
}
pub fn raw_value(cfg:&LucyConfig,selected:usize)->String{
    match selected{
        0=>cfg.models.main.clone(),
        1=>cfg.models.hyprfast_command.clone(),
        2=>cfg.models.main_api_key.clone().or_else(|| cfg.models.api_key.clone()).unwrap_or_default(),
        3=>cfg.models.main_base_url.clone().or_else(|| cfg.models.base_url.clone()).unwrap_or_default(),
        4=>cfg.models.cheap_api_key.clone().or_else(|| cfg.models.api_key.clone()).unwrap_or_default(),
        5=>cfg.models.cheap_base_url.clone().or_else(|| cfg.models.base_url.clone()).unwrap_or_default(),
        12=>cfg.voice.model.clone(),
        13=>cfg.voice.api_key.clone().unwrap_or_default(),
        14=>cfg.voice.push_to_talk.clone(),
        _=>String::new(),
    }
}
