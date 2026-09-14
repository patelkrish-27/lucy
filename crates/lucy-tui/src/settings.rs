use lucy_config::LucyConfig;
use ratatui::{layout::{Alignment,Constraint,Direction,Layout,Rect},style::{Modifier,Style},text::{Line,Span},widgets::{Block,Borders,List,ListItem,Paragraph},Frame};

pub const ITEMS: [&str; 14] = [
    "Main model","HyprFast command model","LLM API Key (any brand)","LLM Endpoint / Base URL",
    "Planner max subtasks","Planner max depth","HyprFast candidates","HyprFast batching",
    "HyprFast parallel","Verify actions","Voice model","Voice API Key","Voice hotkey","Session resume"
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
    let block=Block::default().title(" Lucy Settings — any LLM brand via API Key + Endpoint ").borders(Borders::ALL);
    frame.render_widget(block.clone(),popup);
    let inner=block.inner(popup);
    let cols=Layout::default().direction(Direction::Horizontal).constraints([Constraint::Percentage(45),Constraint::Percentage(55)]).split(inner);
    let items=ITEMS.iter().enumerate().map(|(i,s)|ListItem::new(Line::from(vec![if i==selected{Span::raw("▶ ")}else{Span::raw("  ")},Span::raw(*s)]))).collect::<Vec<_>>();
    frame.render_widget(List::new(items),cols[0]);
    let values=[
        cfg.models.main.clone(),
        cfg.models.hyprfast_command.clone(),
        mask(&cfg.models.api_key),
        cfg.models.base_url.clone().unwrap_or_else(||"https://api.openai.com/v1 (default)".into()),
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
    let detail=Paragraph::new(values[selected].clone()).alignment(Alignment::Center).block(Block::default().title(" Value ").borders(Borders::ALL));
    frame.render_widget(detail,cols[1]);
    let help=Paragraph::new("↑/↓ select  ←/→ change  type to edit  Backspace  Esc close  Ctrl+S save").alignment(Alignment::Center).style(Style::default().add_modifier(Modifier::DIM));
    let help_area=Rect{x:popup.x+1,y:popup.bottom().saturating_sub(2),width:popup.width.saturating_sub(2),height:1};
    frame.render_widget(help,help_area);
}

pub fn change(cfg:&mut LucyConfig, selected:usize, direction:i8) {
    match selected {
        4=>{let v=cfg.planner.max_subtasks as i32+direction as i32;cfg.planner.max_subtasks=v.clamp(1,256) as usize;}
        5=>{let v=cfg.planner.max_depth as i32+direction as i32;cfg.planner.max_depth=v.clamp(1,64) as usize;}
        6=>{let v=cfg.hyprfast.max_candidates as i32+direction as i32;cfg.hyprfast.max_candidates=v.clamp(1,68) as usize;}
        7=>cfg.hyprfast.batching=!cfg.hyprfast.batching,
        8=>cfg.hyprfast.parallel=!cfg.hyprfast.parallel,
        9=>cfg.hyprfast.verify_actions=!cfg.hyprfast.verify_actions,
        13=>cfg.sessions.resume=!cfg.sessions.resume,
        _=>{}
    }
}
pub fn edit_text(cfg:&mut LucyConfig,selected:usize,value:&str){
    let v=value.trim().to_owned();
    match selected{
        0=>cfg.models.main=v,
        1=>cfg.models.hyprfast_command=v,
        2=>cfg.models.api_key=if v.is_empty()||v=="(not set)"{None}else{Some(v)},
        3=>cfg.models.base_url=if v.is_empty()||v=="(not set)"||v.contains("(default)"){None}else{Some(v)},
        10=>cfg.voice.model=v,
        11=>cfg.voice.api_key=if v.is_empty()||v=="(not set)"{None}else{Some(v)},
        12=>cfg.voice.push_to_talk=v,
        _=>{}
    }
}
pub fn raw_value(cfg:&LucyConfig,selected:usize)->String{
    match selected{
        0=>cfg.models.main.clone(),
        1=>cfg.models.hyprfast_command.clone(),
        2=>cfg.models.api_key.clone().unwrap_or_default(),
        3=>cfg.models.base_url.clone().unwrap_or_default(),
        10=>cfg.voice.model.clone(),
        11=>cfg.voice.api_key.clone().unwrap_or_default(),
        12=>cfg.voice.push_to_talk.clone(),
        _=>String::new(),
    }
}
