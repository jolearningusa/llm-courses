/** Actual pinned OpenClaw bundled registry controls; callbacks are course fixtures. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {registerHooks,syncBuiltinESMExports} from 'node:module';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';

const here=path.dirname(fileURLToPath(import.meta.url));
const protocolPath=path.join(here,'openclaw_context_protocol.json');
const protocol=JSON.parse(fs.readFileSync(protocolPath));
const [version,packageRoot,output]=process.argv.slice(2);
if(!protocol.cohorts[version] || !packageRoot || !output)throw Error('version packageRoot new-output.json required');
if(fs.existsSync(output))throw Error('Preserve existing evidence; choose a new output');
assert.equal(process.versions.node,protocol.runtime_node);
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const cohort=protocol.cohorts[version];
for(const [name,expected] of Object.entries(cohort.source_sha256))assert.equal(hash(path.join(packageRoot,name)),expected,'changed packaged file '+name);
assert.ok(process.env.OPENCLAW_STATE_DIR?.startsWith('/private/tmp/'),'private state directory required');
let networkAttempts=0;
const deny=()=>{networkAttempts++;throw Error('External network disabled in context-registry fixture');};
globalThis.fetch=deny;http.request=deny;http.get=deny;https.request=deny;https.get=deny;
net.connect=deny;net.createConnection=deny;net.Socket.prototype.connect=deny;syncBuiltinESMExports();
const resolved=new Set();
const hooks=registerHooks({resolve(specifier,context,next){const result=next(specifier,context);if(result.url.startsWith('file:'))resolved.add(result.url);return result;}});
const started=performance.now();
const results=[];const events=[];
const report={version,node:process.version,platform:process.platform,arch:process.arch,
  protocol_sha256:hash(protocolPath),runner_sha256:hash(fileURLToPath(import.meta.url)),
  npm_integrity:cohort.npm_integrity,tarball_sha256:cohort.tarball_sha256,
  callback_scope:protocol.callbacks,state_directory:process.env.OPENCLAW_STATE_DIR,cases:results,events,
  imported_module_scope:'main-process resolution only; dynamically spawned health-store worker closure is not instrumented',
  full_gateway_or_model_compaction:'not_run',full_installation_qualification:'not_run'};
let runtime,registry;
try{
  registry=await import(pathToFileURL(path.join(packageRoot,cohort.registry_module)));
  runtime=await import(pathToFileURL(path.join(packageRoot,cohort.runtime_module)));
  const empty=await import(pathToFileURL(path.join(packageRoot,cohort.empty_registry_module)));
  assert.equal(registry.o.name,'resolveContextEngine');assert.equal(registry.a.name,'registerContextEngineInRegistry');
  assert.equal(runtime.C.name,'setActivePluginRegistry');assert.equal(empty.t.name,'createEmptyPluginRegistry');
  const active=empty.t();runtime.C(active,'course-context-fixture');
  const owner=id=>'plugin:'+id;
  const config=id=>({plugins:{enabled:true,slots:{contextEngine:id},entries:{[id]:{enabled:true}}}});
  const snapshots=[];
  function engine(id,opts={}){
    const state={id,ingest:0,assemble:0,compact:0,dispose:0,seen:[],memory:[]};snapshots.push(state);
    return {info:{id,name:'Course callback '+id,ownsCompaction:true,...opts.info},
      async ingest(params){state.ingest++;state.memory.push(params.message);return {ingested:true};},
      async assemble(params){state.assemble++;state.seen.push(params);if(opts.assemble)return opts.assemble(params,state);return {messages:params.messages??[],estimatedTokens:0,marker:id};},
      async compact(params){state.compact++;state.seen.push(params);if(opts.compact)return opts.compact(params,state);return {ok:true,compacted:true,summary:state.memory.join('|'),marker:id};},
      async dispose(){state.dispose++;},state};
  }
  const fallbackStates=[];
  const fallbackFactory=()=>{const e=engine('legacy');fallbackStates.push(e.state);return e;};
  assert.deepEqual(registry.a(active,'legacy',fallbackFactory,'core'),{ok:true});
  async function register(id,factory,options){const result=registry.a(active,id,factory,owner(id),options);assert.equal(result.ok,true);}
  const resolve=id=>registry.o(config(id));
  async function quarantines(){return (await registry.p()).map(q=>({engineId:q.engineId,operation:q.operation,reason:q.reason}));}
  async function check(name,body){const tick=performance.now();const evidence=await body();results.push({case:name,outcome:'passed',seconds:(performance.now()-tick)/1000,evidence});}
  await check('reserved_owner_and_duplicate_guard',async()=>{
    const reserved=registry.a(active,'legacy',()=>engine('fake'),'plugin:intruder');assert.equal(reserved.ok,false);assert.equal(reserved.existingOwner,'core');
    await register('owner-control',()=>engine('owner-control'));
    const duplicate=registry.a(active,'owner-control',()=>engine('owner-control'),'plugin:other');assert.equal(duplicate.ok,false);
    const sameOwner=registry.a(active,'owner-control',()=>engine('owner-control'),owner('owner-control'));assert.equal(sameOwner.ok,false);
    return {reserved,duplicate,sameOwner};
  });
  await check('accepted_host_param_projection',async()=>{
    let raw;await register('projection',()=>raw=engine('projection',{info:{acceptedHostParams:['sessionKey']}}));const e=await resolve('projection');
    const controller=new AbortController();const supplied={messages:[{role:'user',content:'fixture'}],sessionKey:'agent:course:one',prompt:'synthetic-hidden-prompt',runtimeSettings:{marker:'synthetic'},sessionTarget:{sessionId:'one'},runtimeContext:{marker:'synthetic'},abortSignal:controller.signal,customMarker:'ordinary-input'};
    await e.assemble(supplied);const assemble=raw.state.seen.at(-1);assert.deepEqual(Object.keys(assemble).sort(),['customMarker','messages','sessionKey']);
    await e.compact(supplied);const compact=raw.state.seen.at(-1);assert.equal(compact.abortSignal,controller.signal);assert.equal(compact.sessionKey,supplied.sessionKey);assert.ok(!('prompt'in compact));
    await e.dispose();return {assemble_keys:Object.keys(assemble).sort(),compact_keys:Object.keys(compact).sort(),compact_abort_signal_same_object:true};
  });
  await check('fresh_factory_state_isolation',async()=>{
    const made=[];await register('isolated',()=>{const e=engine('isolated');made.push(e);return e;});const a=await resolve('isolated'),b=await resolve('isolated');
    await a.ingest({message:'session-A-only'});assert.equal((await a.compact({})).summary,'session-A-only');assert.equal((await b.compact({})).summary,'');assert.equal(made.length,2);assert.notEqual(made[0],made[1]);
    await a.dispose();await b.dispose();return {factory_instances:made.length,first_memory:made[0].state.memory,second_memory:made[1].state.memory,scope:'fresh registry factories, not Gateway session cloning'};
  });
  await check('assembly_error_quarantine_and_fallback',async()=>{
    let calls=0,factories=0;await register('assembly-fail',()=>{factories++;return engine('assembly-fail',{assemble(){calls++;throw Error('course-assembly-failure');}});});
    const e=await resolve('assembly-fail');const first=await e.assemble({messages:[]});assert.equal(first.marker,'legacy');assert.equal((await e.assemble({messages:[]})).marker,'legacy');
    const retried=await resolve('assembly-fail');assert.equal(retried.info.id,'legacy');assert.equal(calls,1);assert.equal(factories,1);
    const q=(await quarantines()).find(q=>q.engineId==='assembly-fail');assert.equal(q.operation,'assemble');await e.dispose();await retried.dispose();return {raw_calls:calls,factory_calls:factories,first_result:first,quarantine:q};
  });
  await check('compaction_error_rethrow_then_fallback_and_explicit_recovery',async()=>{
    const sentinel=Error('course-compaction-failure');let calls=0;await register('compact-fail',()=>engine('compact-fail',{compact(){calls++;throw sentinel;}}));const e=await resolve('compact-fail');
    let caught;try{await e.compact({});}catch(error){caught=error;}assert.equal(caught,sentinel);
    const after=await e.compact({});assert.equal(after.marker,'legacy');assert.equal(calls,1);const q=(await quarantines()).find(q=>q.engineId==='compact-fail');assert.equal(q.operation,'compact');await e.dispose();
    const replaced=await registry.i('compact-fail',()=>engine('compact-fail'),owner('compact-fail'),{allowSameOwnerRefresh:true});assert.equal(replaced.ok,true);assert.ok(!(await quarantines()).some(q=>q.engineId==='compact-fail'));
    const recovered=await resolve('compact-fail');assert.equal((await recovered.compact({})).marker,'compact-fail');await recovered.dispose();return {first_error:caught.message,first_error_identity_preserved:true,raw_calls:calls,next_result:after,quarantine_before_recovery:q,explicit_reregistration_recovered:true};
  });
  await check('invalid_factory_contract',async()=>{
    await register('invalid-contract',()=>({info:{id:'invalid-contract',name:'course'},ingest(){},assemble(){}}));const e=await resolve('invalid-contract');assert.equal(e.info.id,'legacy');
    const q=(await quarantines()).find(q=>q.engineId==='invalid-contract');assert.equal(q.operation,'contract-validation');assert.match(q.reason,/missing compact/);await e.dispose();return {fallback_id:e.info.id,quarantine:q};
  });
  await check('abort_before_call_no_quarantine',async()=>{
    let calls=0;await register('aborted-before',()=>engine('aborted-before',{assemble(){calls++;return {messages:[]};}}));const e=await resolve('aborted-before');const c=new AbortController(),sentinel=Error('course-owner-abort-before');c.abort(sentinel);
    let caught;try{await e.assemble({messages:[],abortSignal:c.signal});}catch(error){caught=error;}assert.equal(caught,sentinel);assert.equal(calls,0);assert.ok(!(await quarantines()).some(q=>q.engineId==='aborted-before'));await e.dispose();return {raw_calls:calls,error:caught.message,same_error_object:true,quarantined:false};
  });
  await check('abort_during_compact_no_quarantine_and_retry',async()=>{
    const c=new AbortController(),sentinel=Error('course-owner-abort-during');let calls=0;await register('aborted-during',()=>engine('aborted-during',{info:{acceptedHostParams:[]},compact(params){calls++;if(calls===1){assert.equal(params.abortSignal,c.signal);c.abort(sentinel);throw sentinel;}return {ok:true,compacted:true,marker:'healthy-after-owner-abort'};}}));
    const e=await resolve('aborted-during');let caught;try{await e.compact({abortSignal:c.signal});}catch(error){caught=error;}assert.equal(caught,sentinel);assert.ok(!(await quarantines()).some(q=>q.engineId==='aborted-during'));
    const retry=await e.compact({});assert.equal(retry.marker,'healthy-after-owner-abort');assert.equal(calls,2);await e.dispose();return {raw_calls:calls,error:caught.message,quarantined:false,retry};
  });
  await check('discovery_only_fallback_without_quarantine',async()=>{
    let factories=0;await register('discovery-only',()=>{factories++;return engine('discovery-only');},{lifecycle:'readOnlyDiscovery'});const e=await resolve('discovery-only');assert.equal(e.info.id,'legacy');assert.equal(factories,0);assert.ok(!(await quarantines()).some(q=>q.engineId==='discovery-only'));await e.dispose();return {factory_calls:factories,fallback_id:e.info.id,quarantined:false};
  });
  await check('idempotent_disposal',async()=>{
    let raw;await register('cleanup',()=>raw=engine('cleanup'));const e=await resolve('cleanup');await e.dispose();await e.dispose();assert.equal(raw.state.dispose,1);return {raw_disposals:raw.state.dispose};
  });
  assert.deepEqual(results.map(r=>r.case),protocol.cases);assert.equal(networkAttempts,0);
  report.final_quarantines=await quarantines();report.status='passed';
}catch(error){report.status='failed';report.error={name:error.name,message:error.message,stack:error.stack};process.exitCode=1;}
finally{
  if(runtime){try{await runtime.r();}catch(error){report.cleanup_error=String(error);process.exitCode=1;report.status='failed';}}
  hooks.deregister();report.network_attempts=networkAttempts;report.seconds=(performance.now()-started)/1000;
  report.main_process_resolved_files=[...resolved].sort().map(url=>{const p=fileURLToPath(url);return {path:p,sha256:hash(p),bytes:fs.statSync(p).size};});
  fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({version,status:report.status,case_groups:results.length,network_attempts:networkAttempts,main_process_files:report.main_process_resolved_files.length,error:report.error?.message}));
}
