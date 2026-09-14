use std::{collections::HashSet, sync::Arc};
use anyhow::Result;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use tokio::sync::mpsc;
pub mod provider;
pub use provider::OpenAIProvider;
const MAX_TOOL_TURNS:usize=64;
pub struct Agent<P:ModelProvider>{provider:Arc<P>,tools:Arc<ToolRegistry>,approver:Option<ApprovalGate>}
impl<P:ModelProvider+'static> Agent<P>{
 pub fn new(provider:Arc<P>,tools:Arc<ToolRegistry>)->Self{Self{provider,tools,approver:None}}
 pub fn with_approval(self,gate:ApprovalGate)->Self{Self{provider:self.provider,tools:self.tools,approver:Some(gate)}}
 pub async fn execute(&self,prompt:String,working_dir:Option<std::path::PathBuf>,interrupt:InterruptSignal)->Result<mpsc::UnboundedReceiver<AgentEvent>>{self.execute_with_history(prompt,Vec::new(),working_dir,interrupt).await}
 pub async fn execute_with_history(&self,prompt:String,history:Vec<TurnMessage>,working_dir:Option<std::path::PathBuf>,interrupt:InterruptSignal)->Result<mpsc::UnboundedReceiver<AgentEvent>>{self.execute_with_history_filtered(prompt,history,working_dir,interrupt,HashSet::new()).await}
  pub async fn execute_with_history_filtered(&self,prompt:String,history:Vec<TurnMessage>,working_dir:Option<std::path::PathBuf>,interrupt:InterruptSignal,allowed_hyprfast:HashSet<String>)->Result<mpsc::UnboundedReceiver<AgentEvent>>{let(provider,tools,approver)=(self.provider.clone(),self.tools.clone(),self.approver.clone());let(tx,rx)=mpsc::unbounded_channel();tokio::spawn(async move{if let Err(err)=run_loop(provider,tools,prompt,history,working_dir,interrupt,tx.clone(),allowed_hyprfast,approver).await{let _=tx.send(AgentEvent::Error{message:err.to_string()});}let _=tx.send(AgentEvent::Done);});Ok(rx)}
}
async fn run_loop<P:ModelProvider>(provider:Arc<P>,tools:Arc<ToolRegistry>,prompt:String,mut history:Vec<TurnMessage>,working_dir:Option<std::path::PathBuf>,interrupt:InterruptSignal,tx:mpsc::UnboundedSender<AgentEvent>,allowed_hyprfast:HashSet<String>,approver:Option<ApprovalGate>)->Result<()>{
 let session_id=SessionId::default();let user=TurnMessage::User(prompt.clone());history.push(user.clone());let _=tx.send(AgentEvent::History{message:user});let mut turns=0;
 loop{if interrupt.is_set(){return Err(LucyError::Cancelled.into())}if turns>=MAX_TOOL_TURNS{return Err(anyhow::anyhow!("tool-turn limit reached ({MAX_TOOL_TURNS})"))}turns+=1;let _=tx.send(AgentEvent::Status{message:format!("Planning turn {turns}…")});
 let request=ModelRequest{session_id:session_id.clone(),prompt:prompt.clone(),history:history.clone(),tools:tools.definitions_filtered(&allowed_hyprfast)};let turn=provider.run_turn(request,tx.clone(),interrupt.clone()).await?;
 if let Some(text)=turn.text.clone(){let _=tx.send(AgentEvent::TextDelta{text});}let assistant=TurnMessage::Assistant(AssistantTurn{text:turn.text.clone(),tool_calls:turn.tool_calls.clone()});history.push(assistant.clone());let _=tx.send(AgentEvent::History{message:assistant});if turn.tool_calls.is_empty()||turn.stop{break}
   for call in turn.tool_calls {
       if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
       if let Some(gate)=approver.as_ref(){
           if gate.needs_approval(&call.name,tools.requires_approval(&call.name)){
               match gate.ask(&call.id,&call.name,&call.input).await{
                   ApprovalDecision::AllowOnce|ApprovalDecision::AllowAlways=>{}
                   ApprovalDecision::Deny=>{
                       let output=serde_json::json!({"denied by user":call.name.clone()});
                       let _=tx.send(AgentEvent::ToolFinished{id:call.id.clone(),name:call.name.clone(),output:output.clone(),is_error:true});
                       let tool=TurnMessage::Tool(ToolResult{call_id:call.id,name:call.name,output,is_error:true});
                       history.push(tool.clone());
                       let _=tx.send(AgentEvent::History{message:tool});
                       continue;
                   }
               }
           }
       }
       let _=tx.send(AgentEvent::ToolStarted{id:call.id.clone(),name:call.name.clone(),input:call.input.clone()});
       let ctx=ToolContext{session_id:session_id.clone(),tool_call_id:call.id.clone(),working_dir:working_dir.clone(),execution_mode:ExecutionMode::Agent,events:tx.clone(),interrupt:interrupt.clone()};
       let result=tools.execute(&call.name,call.input.clone(),ctx).await;
       let(output,is_error)=match result{Ok(v)=>(v,false),Err(e)=>(serde_json::json!({"error":e.to_string()}),true)};
       let truncated_output=truncate_tool_output(&output,80);
       let _=tx.send(AgentEvent::ToolFinished{id:call.id.clone(),name:call.name.clone(),output:truncated_output.clone(),is_error});
       let tool=TurnMessage::Tool(ToolResult{call_id:call.id,name:call.name,output:truncated_output,is_error});
       history.push(tool.clone());
       let _=tx.send(AgentEvent::History{message:tool});
   }
  }
  Ok(())
}
