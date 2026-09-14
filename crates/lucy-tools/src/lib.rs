use std::{collections::{HashMap, HashSet}, sync::Arc, time::Duration};
use anyhow::{anyhow, Result};
use lucy_core::*;
use serde_json::Value;
use tokio::io::AsyncReadExt;
const DEFAULT_OUTPUT_LIMIT: usize = 64 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
pub struct ToolRegistry { tools: HashMap<String, Arc<dyn Tool>> }
impl ToolRegistry {
 pub fn new()->Self{Self{tools:HashMap::new()}}
 pub fn register<T:Tool+'static>(&mut self,tool:T){self.tools.insert(tool.name().to_string(),Arc::new(tool));}
 pub fn register_arc(&mut self,tool:Arc<dyn Tool>){self.tools.insert(tool.name().to_string(),tool);}
 pub fn get(&self,name:&str)->Option<Arc<dyn Tool>>{self.tools.get(name).cloned()}
 pub fn requires_approval(&self,name:&str)->bool{self.tools.get(name).map(|t|t.requires_approval()).unwrap_or(true)}
 pub fn definitions(&self)->Vec<Value>{self.tools.values().map(|t|serde_json::json!({"name":t.name(),"description":t.description(),"input_schema":t.parameters_schema()})).collect()}
 pub fn definitions_filtered(&self,allowed:&HashSet<String>)->Vec<Value>{self.tools.values().filter(|t|!t.name().starts_with("mcp_hyprfast_")||allowed.contains(t.name())).map(|t|serde_json::json!({"name":t.name(),"description":t.description(),"input_schema":t.parameters_schema()})).collect()}
 pub fn definitions_for_names(&self,allowed:&HashSet<String>)->Vec<Value>{self.tools.values().filter(|t|allowed.contains(t.name())).map(|t|serde_json::json!({"name":t.name(),"description":t.description(),"input_schema":t.parameters_schema()})).collect()}
 /// Names of built-in local tools (everything not served over MCP).
 /// Used by the hierarchical planner for `files`/`shell` subtasks so the
 /// cheap command model can act on the local machine, not just the desktop.
 pub fn local_tool_names(&self)->HashSet<String>{self.tools.values().filter(|t|!t.name().starts_with("mcp_")).map(|t|t.name().to_string()).collect()}
 pub async fn execute(&self,name:&str,input:Value,ctx:ToolContext)->Result<Value>{if ctx.interrupt.is_set(){return Err(LucyError::Cancelled.into())}let tool=self.get(name).ok_or_else(||anyhow!(LucyError::ToolNotFound(name.to_string())))?;tool.execute(input,ctx).await}
}
impl Default for ToolRegistry{fn default()->Self{Self::new()}}
fn allowed_command(command:&str)->bool{if std::env::var("LUCY_ALLOW_DANGEROUS").ok().as_deref()==Some("1"){return true;}let c=command.to_ascii_lowercase();let blocked=["rm -rf /","mkfs","dd if=",":(){:|:&};:","shutdown","reboot","poweroff","chmod -r 777 /","chown -r"];!blocked.iter().any(|x|c.contains(x))}
pub struct ShellTool{pub output_limit:usize,pub timeout:Duration}
impl Default for ShellTool{fn default()->Self{Self{output_limit:DEFAULT_OUTPUT_LIMIT,timeout:DEFAULT_TIMEOUT}}}
#[async_trait::async_trait]
impl Tool for ShellTool{
 fn name(&self)->&str{"shell"}
 fn description(&self)->&str{"Run a shell command in Lucy's working directory. Destructive system-wide commands are blocked unless LUCY_ALLOW_DANGEROUS=1."}
 fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"},"intent":{"type":"string","description":"Why this tool call is needed"}},"required":["command","intent"]})}
 async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let command=input.get("command").and_then(Value::as_str).ok_or_else(||anyhow!(LucyError::InvalidInput("command is required".into())))?;if !allowed_command(command){return Err(anyhow!("command blocked by Lucy safety policy; set LUCY_ALLOW_DANGEROUS=1 only when explicitly requested"));}let working_dir=ctx.working_dir.clone().unwrap_or(std::env::current_dir()?);let mut child=tokio::process::Command::new("sh").arg("-lc").arg(command).current_dir(&working_dir).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn()?;let stdout=child.stdout.take().ok_or_else(||anyhow!("failed to capture stdout"))?;let stderr=child.stderr.take().ok_or_else(||anyhow!("failed to capture stderr"))?;let limit=self.output_limit;let out_task=tokio::spawn(async move{read_limited(stdout,limit).await});let err_task=tokio::spawn(async move{read_limited(stderr,limit).await});tokio::select!{status=child.wait()=>{let status=status?;let stdout=out_task.await??;let stderr=err_task.await??;Ok(serde_json::json!({"status":status.code(),"success":status.success(),"stdout":stdout,"stderr":stderr,"truncated":stdout.len()>=limit||stderr.len()>=limit}))},_=ctx.interrupt.notified()=>{let _=child.kill().await;Err(LucyError::Cancelled.into())},_=tokio::time::sleep(self.timeout)=>{let _=child.kill().await;Err(anyhow!("shell tool timed out after {} seconds",self.timeout.as_secs()))}}}
}
pub struct ReadFileTool;
#[async_trait::async_trait] impl Tool for ReadFileTool{fn name(&self)->&str{"read_file"}fn description(&self)->&str{"Read a UTF-8 text file."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})}fn requires_approval(&self)->bool{false}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let p=input.get("path").and_then(Value::as_str).ok_or_else(||anyhow!("path is required"))?;let path=ctx.resolve_path(std::path::Path::new(p));let content=tokio::fs::read_to_string(&path).await?;Ok(serde_json::json!({"path":path,"content":content}))}}
pub struct WriteFileTool;
#[async_trait::async_trait] impl Tool for WriteFileTool{fn name(&self)->&str{"write_file"}fn description(&self)->&str{"Create or replace a UTF-8 text file."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]})}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let p=input.get("path").and_then(Value::as_str).ok_or_else(||anyhow!("path is required"))?;let content=input.get("content").and_then(Value::as_str).ok_or_else(||anyhow!("content is required"))?;let path=ctx.resolve_path(std::path::Path::new(p));if let Some(parent)=path.parent(){tokio::fs::create_dir_all(parent).await?;}tokio::fs::write(&path,content).await?;Ok(serde_json::json!({"path":path,"bytes":content.len()}))}}
pub struct ListDirTool;
#[async_trait::async_trait] impl Tool for ListDirTool{fn name(&self)->&str{"list_dir"}fn description(&self)->&str{"List files and directories."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})}fn requires_approval(&self)->bool{false}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let p=input.get("path").and_then(Value::as_str).unwrap_or(".");let path=ctx.resolve_path(std::path::Path::new(p));let mut rd=tokio::fs::read_dir(&path).await?;let mut items=Vec::new();while let Some(e)=rd.next_entry().await?{let ft=e.file_type().await?;items.push(serde_json::json!({"name":e.file_name().to_string_lossy(),"directory":ft.is_dir()}));}Ok(serde_json::json!({"path":path,"entries":items}))}}
pub struct SearchFilesTool;
#[async_trait::async_trait] impl Tool for SearchFilesTool{fn name(&self)->&str{"search_files"}fn description(&self)->&str{"Search text recursively with ripgrep."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"query":{"type":"string"},"path":{"type":"string"}},"required":["query"]})}fn requires_approval(&self)->bool{false}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let query=input.get("query").and_then(Value::as_str).ok_or_else(||anyhow!("query is required"))?;let p=input.get("path").and_then(Value::as_str).unwrap_or(".");let dir=ctx.resolve_path(std::path::Path::new(p));let o=tokio::process::Command::new("rg").arg("--line-number").arg("--hidden").arg("--glob").arg("!.git").arg(query).arg(&dir).output().await?;Ok(serde_json::json!({"success":o.status.success(),"stdout":String::from_utf8_lossy(&o.stdout),"stderr":String::from_utf8_lossy(&o.stderr)}))}}
pub struct GitTool;
#[async_trait::async_trait] impl Tool for GitTool{fn name(&self)->&str{"git"}fn description(&self)->&str{"Run a git command in the working directory."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"args":{"type":"array","items":{"type":"string"}}},"required":["args"]})}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let args=input.get("args").and_then(Value::as_array).ok_or_else(||anyhow!("args is required"))?;let args:Vec<String>=args.iter().filter_map(Value::as_str).map(str::to_owned).collect();let dir=ctx.working_dir.clone().unwrap_or(std::env::current_dir()?);let o=tokio::process::Command::new("git").args(&args).current_dir(dir).output().await?;Ok(serde_json::json!({"success":o.status.success(),"stdout":String::from_utf8_lossy(&o.stdout),"stderr":String::from_utf8_lossy(&o.stderr),"status":o.status.code()}))}}
pub struct EditFileTool;
#[async_trait::async_trait] impl Tool for EditFileTool{fn name(&self)->&str{"edit_file"}fn description(&self)->&str{"Replace text in a UTF-8 text file (exact match). Set replace_all=true to replace every occurrence."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]})}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let p=input.get("path").and_then(Value::as_str).ok_or_else(||anyhow!(LucyError::InvalidInput("path is required".into())))?;let old=input.get("old_string").and_then(Value::as_str).ok_or_else(||anyhow!(LucyError::InvalidInput("old_string is required".into())))?;let new=input.get("new_string").and_then(Value::as_str).ok_or_else(||anyhow!(LucyError::InvalidInput("new_string is required".into())))?;let replace_all=input.get("replace_all").and_then(Value::as_bool).unwrap_or(false);if old.is_empty(){return Err(anyhow!(LucyError::InvalidInput("old_string must not be empty".into())));}let path=ctx.resolve_path(std::path::Path::new(p));let content=tokio::fs::read_to_string(&path).await?;let count=content.matches(old).count();if count==0{return Err(anyhow!(LucyError::InvalidInput("old_string not found in file".into())));}if count>1&&!replace_all{return Err(anyhow!(LucyError::InvalidInput(format!("old_string matches {count} times; be more specific or set replace_all=true"))));}let replacements=count as u64;let updated=if replace_all{content.replace(old,new)}else{content.replacen(old,new,1)};let preview=edit_preview(&content,old,new,replace_all);tokio::fs::write(&path,&updated).await?;Ok(serde_json::json!({"path":path,"replacements":if replace_all{replacements}else{1},"diff":preview}))}}
fn edit_preview(orig:&str,old:&str,new:&str,replace_all:bool)->String{
 let lines:Vec<&str>=orig.lines().collect();let n=lines.len();
 let mut starts:Vec<usize>=Vec::new();
 if replace_all{let mut from=0usize;while let Some(i)=orig[from..].find(old){let idx=from+i;starts.push(orig[..idx].matches('\n').count());from=idx+old.len().max(1);if from>=orig.len(){break}}}
 let covered:Vec<(usize,usize)>=if replace_all{starts.iter().map(|&s|{let e=line_of_pos(orig,s,old);(s,e)}).collect()}else{let idx=orig.find(old).unwrap_or(0);let s=orig[..idx].matches('\n').count();vec![(s,line_of_pos(orig,s,old))]};
 fn line_of_pos(orig:&str,start_line:usize,old:&str)->usize{let extra=old.matches('\n').count();let mut e=start_line+extra;if old.ends_with('\n'){e=e.saturating_sub(1);}let max=orig.lines().count().saturating_sub(1);e.min(max)}
 // merge covered ranges (with 3-line context windows) into hunks
 let mut merged:Vec<(usize,usize)>=Vec::new();
 let mut sorted=covered.clone();sorted.sort();
 for (s,e) in sorted{let s=s.min(n.saturating_sub(1));let e=e.min(n.saturating_sub(1)).max(s);if let Some(last)=merged.last_mut(){if s<=last.1+7{last.1=last.1.max(e);continue}}merged.push((s,e));}
 // byte offsets of each line start for slice replacement
 let mut offsets:Vec<usize>=Vec::with_capacity(n+1);let mut o=0usize;for l in &lines{offsets.push(o);o+=l.len()+1;}
 let mut out:Vec<String>=Vec::new();
 for (cs,ce) in merged{
  let hs=cs.saturating_sub(3);let he=(ce+3).min(n.saturating_sub(1));
  let slice_start=*offsets.get(cs).unwrap_or(&0);let slice_end=if ce+1<n{*offsets.get(ce+1).unwrap_or(&orig.len())}else{orig.len()};
  let slice=orig.get(slice_start..slice_end).unwrap_or("");
  let new_slice=if replace_all{slice.replace(old,new)}else{slice.replacen(old,new,1)};
  for l in hs..cs{if let Some(t)=lines.get(l){out.push(format!(" {t}"));}}
  for l in cs..=ce{if let Some(t)=lines.get(l){out.push(format!("-{t}"));}}
  for t in new_slice.lines(){out.push(format!("+{t}"));}
  for l in ce+1..=he{if let Some(t)=lines.get(l){out.push(format!(" {t}"));}}
  if out.len()>=60{break}
 }
 if out.len()>60{out.truncate(60);}
 out.join("\n")
}
pub struct GlobTool;
#[async_trait::async_trait] impl Tool for GlobTool{fn name(&self)->&str{"glob"}fn description(&self)->&str{"Find files by glob pattern (*, ?, ** segments). Recursive; skips .git, target and hidden directories; does not follow symlinked directories."}fn parameters_schema(&self)->Value{serde_json::json!({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]})}fn requires_approval(&self)->bool{false}async fn execute(&self,input:Value,ctx:ToolContext)->Result<Value>{let pattern=input.get("pattern").and_then(Value::as_str).ok_or_else(||anyhow!(LucyError::InvalidInput("pattern is required".into())))?;if pattern.is_empty(){return Err(anyhow!(LucyError::InvalidInput("pattern must not be empty".into())));}let p=input.get("path").and_then(Value::as_str).unwrap_or(".");let base=ctx.resolve_path(std::path::Path::new(p));let rels=glob_walk(&base)?;let pat=pattern.strip_prefix("./").unwrap_or(pattern);let segs:Vec<&str>=pat.split('/').collect();let mut matched:Vec<String>=rels.into_iter().filter(|r|glob_path_match(&segs,&r.split('/').collect::<Vec<_>>())).collect();matched.sort();let truncated=matched.len()>200;if truncated{matched.truncate(200);}Ok(serde_json::json!({"pattern":pattern,"path":p,"files":matched,"truncated":truncated}))}}
fn glob_walk(base:&std::path::Path)->Result<Vec<String>>{
 let mut files=Vec::new();let mut stack=vec![base.to_path_buf()];
 while let Some(dir)=stack.pop(){
  let entries=match std::fs::read_dir(&dir){Ok(e)=>e,Err(_)=>continue};
  for entry in entries.flatten(){
   let path=entry.path();let name=entry.file_name().to_string_lossy().into_owned();
   let md=match std::fs::symlink_metadata(&path){Ok(m)=>m,Err(_)=>continue};
   if md.file_type().is_symlink(){
    match std::fs::metadata(&path){Ok(t)if t.is_dir()=>continue,Err(_)=>{},_=>{}}
    if let Ok(rel)=path.strip_prefix(base){files.push(rel.to_string_lossy().replace(std::path::MAIN_SEPARATOR,"/"));}
    continue
   }
   if md.is_dir(){
    if name==".git"||name=="target"||name.starts_with('.'){continue}
    stack.push(path);continue
   }
   if let Ok(rel)=path.strip_prefix(base){let s=rel.to_string_lossy().replace(std::path::MAIN_SEPARATOR,"/");if !s.is_empty(){files.push(s);}}
  }
 }
 Ok(files)
}
fn glob_segment_match(pat:&str,text:&str)->bool{
 let (p,t)=(pat.as_bytes(),text.as_bytes());let (mut pi,mut ti)=(0usize,0usize);let (mut star,mut mark):(Option<usize>,usize)= (None,0usize);
 while ti<t.len(){if pi<p.len()&&(p[pi]==b'?'||p[pi]==t[ti]){pi+=1;ti+=1;}else if pi<p.len()&&p[pi]==b'*'{star=Some(pi);mark=ti;pi+=1;}else if star.is_some(){pi=star.unwrap()+1;mark+=1;ti=mark;}else{return false;}}
 while pi<p.len()&&p[pi]==b'*'{pi+=1;}
 pi==p.len()
}
fn glob_path_match(pat:&[&str],path:&[&str])->bool{
 if pat.is_empty(){return path.is_empty();}
 if pat[0]=="**"{for i in 0..=path.len(){if glob_path_match(&pat[1..],&path[i..]){return true;}}return false;}
 if path.is_empty(){return false;}
 glob_segment_match(pat[0],path[0])&&glob_path_match(&pat[1..],&path[1..])
}
pub fn default_registry()->ToolRegistry{let mut r=ToolRegistry::new();r.register(ShellTool::default());r.register(ReadFileTool);r.register(WriteFileTool);r.register(ListDirTool);r.register(SearchFilesTool);r.register(GitTool);r.register(EditFileTool);r.register(GlobTool);r}
async fn read_limited<R:tokio::io::AsyncRead+Unpin>(mut reader:R,limit:usize)->Result<String>{let mut buf=Vec::with_capacity(limit.min(8192));let mut chunk=[0u8;8192];loop{let n=reader.read(&mut chunk).await?;if n==0{break}let remaining=limit.saturating_sub(buf.len());buf.extend_from_slice(&chunk[..n.min(remaining)]);if buf.len()>=limit{break}}Ok(String::from_utf8_lossy(&buf).into_owned())}
