"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[8312],{65597:function(e,n,i){i.d(n,{f:function(){return l}});var t=i(9518),r=i(23831),o=i(49125),u=i(73942),l=t.default.div.withConfig({displayName:"Tablestyle__PopupContainerStyle",componentId:"sc-8ammqd-0"})(["position:absolute;max-height:","px;z-index:10;border-radius:","px;padding:","px;"," "," "," ",""],58*o.iI,u.TR,2*o.iI,(function(e){return"\n    box-shadow: ".concat((e.theme.shadow||r.Z.shadow).popup,";\n    background-color: ").concat((e.theme.interactive||r.Z.interactive).defaultBackground,";\n  ")}),(function(e){return e.leftOffset&&"\n    left: ".concat(e.leftOffset,"px;\n  ")}),(function(e){return e.topOffset&&"\n    top: ".concat(e.topOffset,"px;\n  ")}),(function(e){return e.width&&"\n    width: ".concat(e.width,"px;\n  ")}))},97496:function(e,n,i){var t=i(82394),r=i(75582),o=i(12691),u=i.n(o),l=i(34376),c=i.n(l),s=i(82684),d=i(83455),a=i(60328),p=i(38341),f=i(47999),h=i(93461),x=i(67971),j=i(10919),g=i(47409),b=i(86673),Z=i(54283),m=i(58180),_=i(19711),C=i(82531),O=i(23831),v=i(73942),E=i(10503),I=i(65597),k=i(93348),P=i(45838),w=i(49125),y=i(19395),N=i(24224),D=i(9736),A=i(96510),L=i(28598);function R(e,n){var i=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),i.push.apply(i,t)}return i}function T(e){for(var n=1;n<arguments.length;n++){var i=null!=arguments[n]?arguments[n]:{};n%2?R(Object(i),!0).forEach((function(n){(0,t.Z)(e,n,i[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(i)):R(Object(i)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(i,n))}))}return e}function S(e){var n=e.cancelingRunId,i=e.disabled,t=e.isLoadingCancelPipeline,o=e.onCancel,u=e.onSuccess,l=e.pipelineRun,c=e.setCancelingRunId,p=e.setErrors,h=e.setShowConfirmationId,j=e.showConfirmationId,m=(0,D.Ct)(),P=l||{},y=P.id,N=P.pipeline_schedule_id,R=P.pipeline_schedule_token,T=P.pipeline_schedule_type,S=P.status,V=t&&y===n&&g.VO.RUNNING===S,U=(0,d.Db)(k.Xm.API===T&&R?C.ZP.pipeline_runs.pipeline_schedules.useCreateWithParent(N,R):C.ZP.pipeline_runs.pipeline_schedules.useCreate(N),{onSuccess:function(e){return(0,A.wD)(e,{callback:function(){u()},onErrorCallback:function(e,n){return null===p||void 0===p?void 0:p({errors:n,response:e})}})}}),F=(0,r.Z)(U,1)[0],M=(0,s.useCallback)((function(){h(null),F({pipeline_run:{backfill_id:null===l||void 0===l?void 0:l.backfill_id,execution_date:null===l||void 0===l?void 0:l.execution_date,pipeline_schedule_id:null===l||void 0===l?void 0:l.pipeline_schedule_id,pipeline_uuid:null===l||void 0===l?void 0:l.pipeline_uuid,variables:null===l||void 0===l?void 0:l.variables}})}),[F,l,h]),B=(0,s.useCallback)((function(){h(null),c(y),o({id:y,status:g.VO.CANCELLED})}),[o,y,c,h]);return(0,L.jsxs)("div",{style:{position:"relative"},children:[(0,L.jsx)(a.Z,{backgroundColor:V&&O.Z.accent.yellow,beforeIcon:g.VO.INITIAL!==S&&!i&&(0,L.jsxs)(L.Fragment,{children:[g.VO.COMPLETED===S&&(0,L.jsx)(E.Jr,{size:2*w.iI}),[g.VO.FAILED,g.VO.CANCELLED].includes(S)&&(0,L.jsx)(E.Py,{inverted:g.VO.CANCELLED===S&&!m,size:2*w.iI}),[g.VO.RUNNING].includes(S)&&(0,L.jsx)(Z.Z,{color:V?O.Z.status.negative:O.Z.monotone.white,small:!0})]}),borderRadius:v.D7,danger:g.VO.FAILED===S&&!m,default:g.VO.INITIAL===S,disabled:i||m,loading:!l,onClick:function(){return h(y)},padding:"6px",primary:g.VO.RUNNING===S&&!V&&!m,warning:g.VO.CANCELLED===S&&!m,children:i?"Ready":V?"Canceling":g.Do[S]}),(0,L.jsx)(f.Z,{onClickOutside:function(){return h(null)},open:j===y,children:(0,L.jsxs)(I.f,{children:[[g.VO.RUNNING,g.VO.INITIAL].includes(S)&&(0,L.jsxs)(L.Fragment,{children:[(0,L.jsx)(_.ZP,{bold:!0,color:"#9ECBFF",children:"Run is in progress"}),(0,L.jsx)(b.Z,{mb:1}),(0,L.jsxs)(_.ZP,{children:["This pipeline run is currently ongoing. Retrying will cancel",(0,L.jsx)("br",{}),"the current pipeline run."]}),(0,L.jsx)(_.ZP,{}),(0,L.jsx)(b.Z,{mt:1,children:(0,L.jsxs)(x.Z,{children:[(0,L.jsx)(a.Z,{onClick:function(){B(),M()},children:"Retry run"}),(0,L.jsx)(b.Z,{ml:1}),(0,L.jsx)(a.Z,{onClick:B,children:"Cancel run"})]})})]}),[g.VO.CANCELLED,g.VO.FAILED,g.VO.COMPLETED].includes(S)&&(0,L.jsxs)(L.Fragment,{children:[(0,L.jsxs)(_.ZP,{bold:!0,color:"#9ECBFF",children:["Run ",S]}),(0,L.jsx)(b.Z,{mb:1}),(0,L.jsx)(_.ZP,{children:"Retry the run with changes you have made to the pipeline."}),(0,L.jsx)(b.Z,{mb:1}),(0,L.jsx)(a.Z,{onClick:M,children:"Retry run"})]})]})})]})}n.Z=function(e){var n=e.allowBulkSelect,i=e.disableRowSelect,o=e.emptyMessage,l=void 0===o?"No runs available":o,f=e.fetchPipelineRuns,Z=e.onClickRow,O=e.pipelineRuns,I=e.selectedRun,k=e.selectedRuns,D=e.setSelectedRuns,R=e.setErrors,V=(0,s.useState)(null),U=V[0],F=V[1],M=(0,s.useState)(null),B=M[0],H=M[1],z=(0,d.Db)((function(e){var n=e.id,i=e.status;return C.ZP.pipeline_runs.useUpdate(n)({pipeline_run:{status:i}})}),{onSuccess:function(e){return(0,A.wD)(e,{callback:function(){F(null),f()},onErrorCallback:function(e,n){F(null),null===R||void 0===R||R({errors:n,response:e})}})}}),Y=(0,r.Z)(z,2),G=Y[0],J=Y[1].isLoading,K=[null,1,2,1,1,null],q=[{uuid:"Status"},{uuid:"Pipeline UUID"},{uuid:"Date"},{uuid:"Trigger"},{uuid:"Block runs"},{uuid:"Completed"},{uuid:"Logs"}],Q=(0,s.useMemo)((function(){return O.every((function(e){var n=e.id;return!(null===k||void 0===k||!k[n])}))}),[O,k]);return n&&(K.unshift(null),q.unshift({label:function(){return(0,L.jsx)(p.Z,{checked:Q,onClick:function(){var e=(0,N.HK)(O||[],(function(e){return e.id}));D(Q?{}:e)}})},uuid:"Selected"})),!i&&Z&&(K.push(null),q.push({label:function(){return""},uuid:"action"})),(0,L.jsx)(P.cl,{minHeight:30*w.iI,overflowVisible:!!B,children:0===(null===O||void 0===O?void 0:O.length)?(0,L.jsx)(b.Z,{px:3,py:1,children:(0,L.jsx)(_.ZP,{bold:!0,default:!0,monospace:!0,muted:!0,children:l})}):(0,L.jsx)(m.Z,{columnFlex:K,columns:q,isSelectedRow:function(e){return!i&&O[e].id===(null===I||void 0===I?void 0:I.id)},onClickRow:i?null:Z,rowVerticalPadding:6,rows:null===O||void 0===O?void 0:O.map((function(e,r){var o=e.block_runs_count,l=e.completed_at,s=e.execution_date,d=e.id,m=e.pipeline_schedule_id,C=e.pipeline_schedule_name,I=e.pipeline_uuid,P=e.status,N=!d&&!P,A=[];if(A=r>0&&O[r-1].execution_date===e.execution_date&&O[r-1].pipeline_schedule_id===e.pipeline_schedule_id?[(0,L.jsx)(b.Z,{ml:1,children:(0,L.jsxs)(x.Z,{alignItems:"center",children:[(0,L.jsx)(E.TT,{size:2*w.iI,useStroke:!0}),(0,L.jsx)(a.Z,{borderRadius:v.D7,notClickable:!0,padding:"6px",children:(0,L.jsx)(_.ZP,{muted:!0,children:g.Do[P]})})]})},"row_status"),(0,L.jsx)(_.ZP,{default:!0,monospace:!0,muted:!0,children:I},"row_pipeline_uuid"),(0,L.jsx)(_.ZP,{default:!0,monospace:!0,muted:!0,children:"-"},"row_date_retry"),(0,L.jsx)(_.ZP,{default:!0,monospace:!0,muted:!0,children:"-"},"row_trigger_retry"),(0,L.jsx)(u(),{as:"/pipelines/".concat(I,"/runs/").concat(d),href:"/pipelines/[pipeline]/runs/[run]",passHref:!0,children:(0,L.jsx)(j.Z,{bold:!0,muted:!0,children:"See block runs (".concat(o,")")})},"row_block_runs"),(0,L.jsx)(_.ZP,{monospace:!0,muted:!0,children:l&&(0,y.Vx)(l)||"-"},"row_completed"),(0,L.jsx)(a.Z,{default:!0,iconOnly:!0,noBackground:!0,onClick:function(){return c().push("/pipelines/".concat(I,"/logs?pipeline_run_id[]=").concat(d))},children:(0,L.jsx)(E.B4,{default:!0,size:2*w.iI})},"row_logs")]:[(0,L.jsx)(S,{cancelingRunId:U,disabled:N,isLoadingCancelPipeline:J,onCancel:G,onSuccess:f,pipelineRun:e,setCancelingRunId:F,setErrors:R,setShowConfirmationId:H,showConfirmationId:B},"row_retry_button"),(0,L.jsx)(_.ZP,{default:!0,monospace:!0,children:I},"row_pipeline_uuid"),(0,L.jsx)(_.ZP,{default:!0,monospace:!0,children:s&&(0,y.Vx)(s)||"-"},"row_date"),(0,L.jsx)(u(),{as:"/pipelines/".concat(I,"/triggers/").concat(m),href:"/pipelines/[pipeline]/triggers/[...slug]",passHref:!0,children:(0,L.jsx)(j.Z,{bold:!0,sameColorAsText:!0,children:C})},"row_trigger"),(0,L.jsx)(u(),{as:"/pipelines/".concat(I,"/runs/").concat(d),href:"/pipelines/[pipeline]/runs/[run]",passHref:!0,children:(0,L.jsx)(j.Z,{bold:!0,disabled:N,sameColorAsText:!0,children:N?"":"See block runs (".concat(o,")")})},"row_block_runs"),(0,L.jsx)(_.ZP,{default:!0,monospace:!0,children:l&&(0,y.Vx)(l)||"-"},"row_completed"),(0,L.jsx)(a.Z,{default:!0,disabled:N,iconOnly:!0,noBackground:!0,onClick:function(){return c().push("/pipelines/".concat(I,"/logs?pipeline_run_id[]=").concat(d))},children:(0,L.jsx)(E.B4,{default:!0,size:2*w.iI})},"row_item_13")],n){var V=!(null===k||void 0===k||!k[d]);A.unshift((0,L.jsx)(p.Z,{checked:V,onClick:function(){D((function(n){return T(T({},n),{},(0,t.Z)({},d,V?null:e))}))}},"selected-pipeline-run-".concat(d)))}return!i&&Z&&A.push((0,L.jsx)(h.Z,{flex:1,justifyContent:"flex-end",children:(0,L.jsx)(E._Q,{default:!0,size:2*w.iI})})),A})),uuid:"pipeline-runs"})})}},19395:function(e,n,i){i.d(n,{IJ:function(){return d},Vx:function(){return p},eI:function(){return a},gU:function(){return h},tL:function(){return f},vJ:function(){return x}});var t,r,o=i(82394),u=i(92083),l=i.n(u);function c(e,n){var i=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),i.push.apply(i,t)}return i}function s(e){for(var n=1;n<arguments.length;n++){var i=null!=arguments[n]?arguments[n]:{};n%2?c(Object(i),!0).forEach((function(n){(0,o.Z)(e,n,i[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(i)):c(Object(i)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(i,n))}))}return e}function d(e){return null===e||void 0===e?void 0:e.reduce((function(e,n){var i=n.block_uuid,t=n.completed_at,r=n.started_at,u=n.status,c=null;r&&t&&(c=l()(t).valueOf()-l()(r).valueOf());return s(s({},e),{},(0,o.Z)({},i,{runtime:c,status:u}))}),{})}function a(e){if(!e)return null;var n=new Date(l()(e).valueOf()),i=Date.UTC(n.getFullYear(),n.getMonth(),n.getDate(),n.getHours(),n.getMinutes(),n.getSeconds());return new Date(i)}function p(e){return"string"!==typeof e?e:a(e.split("+")[0]).toISOString().split(".")[0]}!function(e){e.DAY="day",e.HOUR="hour",e.MINUTE="minute",e.SECOND="second"}(r||(r={}));var f=(t={},(0,o.Z)(t,r.DAY,86400),(0,o.Z)(t,r.HOUR,3600),(0,o.Z)(t,r.MINUTE,60),(0,o.Z)(t,r.SECOND,1),t);function h(e){var n=r.SECOND,i=e;return e%86400===0?(i/=86400,n=r.DAY):e%3600===0?(i/=3600,n=r.HOUR):e%60===0&&(i/=60,n=r.MINUTE),{time:i,unit:n}}function x(e,n){return e*f[n]}},51099:function(e,n,i){i.d(n,{Q:function(){return d}});i(82684);var t=i(60328),r=i(67971),o=i(86673),u=i(10503),l=i(73899),c=i(49125),s=i(28598),d=22;n.Z=function(e){var n=e.page,i=e.maxPages,d=e.onUpdate,a=e.totalPages,p=[],f=i;if(f>a)p=Array.from({length:a},(function(e,n){return n}));else{var h=Math.floor(f/2),x=n-h;n+h>=a?(x=a-f+2,f-=2):n-h<=0?(x=0,f-=2):(f-=4,x=n-Math.floor(f/2)),p=Array.from({length:f},(function(e,n){return n+x}))}return(0,s.jsx)(s.Fragment,{children:a>0&&(0,s.jsxs)(r.Z,{alignItems:"center",children:[(0,s.jsx)(t.Z,{disabled:0===n,onClick:function(){return d(n-1)},children:(0,s.jsx)(u.Hd,{size:1.5*c.iI,stroke:"#AEAEAE"})}),!p.includes(0)&&(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(o.Z,{ml:1,children:(0,s.jsx)(t.Z,{onClick:function(){return d(0)},borderLess:!0,noBackground:!0,children:1})},0),!p.includes(1)&&(0,s.jsx)(o.Z,{ml:1,children:(0,s.jsx)(t.Z,{notClickable:!0,noBackground:!0,noPadding:!0,children:"..."})},0)]}),p.map((function(e){return(0,s.jsx)(o.Z,{ml:1,children:(0,s.jsx)(t.Z,{onClick:function(){e!==n&&d(e)},notClickable:e===n,backgroundColor:e===n&&l.a$,borderLess:!0,noBackground:!0,children:e+1})},e)})),!p.includes(a-1)&&(0,s.jsxs)(s.Fragment,{children:[!p.includes(a-2)&&(0,s.jsx)(o.Z,{ml:1,children:(0,s.jsx)(t.Z,{notClickable:!0,noBackground:!0,noPadding:!0,children:"..."})},0),(0,s.jsx)(o.Z,{ml:1,children:(0,s.jsx)(t.Z,{onClick:function(){return d(a-1)},borderLess:!0,noBackground:!0,children:a})},a-1)]}),(0,s.jsx)(o.Z,{ml:1}),(0,s.jsx)(t.Z,{disabled:n===a-1,onClick:function(){return d(n+1)},children:(0,s.jsx)(u.Kw,{size:1.5*c.iI,stroke:"#AEAEAE"})})]})})}},47409:function(e,n,i){i.d(n,{Az:function(){return l},BF:function(){return u},Do:function(){return s},VO:function(){return o},sZ:function(){return c}});var t,r=i(82394),o=i(66050).V,u=[o.INITIAL,o.RUNNING],l=[o.CANCELLED,o.COMPLETED,o.FAILED],c="__mage_variables",s=(t={},(0,r.Z)(t,o.CANCELLED,"Cancelled"),(0,r.Z)(t,o.COMPLETED,"Done"),(0,r.Z)(t,o.FAILED,"Failed"),(0,r.Z)(t,o.INITIAL,"Ready"),(0,r.Z)(t,o.RUNNING,"Running"),t)},93348:function(e,n,i){i.d(n,{TR:function(){return a},U5:function(){return c},Xm:function(){return o},Z4:function(){return d},fq:function(){return l},kJ:function(){return s}});var t,r,o,u=i(82394);!function(e){e.API="api",e.EVENT="event",e.TIME="time"}(o||(o={}));var l,c,s,d=(t={},(0,u.Z)(t,o.API,(function(){return"API"})),(0,u.Z)(t,o.EVENT,(function(){return"event"})),(0,u.Z)(t,o.TIME,(function(){return"schedule"})),t);!function(e){e.ACTIVE="active",e.INACTIVE="inactive"}(l||(l={})),function(e){e.ONCE="@once",e.HOURLY="@hourly",e.DAILY="@daily",e.WEEKLY="@weekly",e.MONTHLY="@monthly"}(c||(c={})),function(e){e.CREATED_AT="created_at",e.NAME="name",e.PIPELINE="pipeline_uuid",e.STATUS="status",e.TYPE="schedule_type"}(s||(s={}));var a=(r={},(0,u.Z)(r,s.CREATED_AT,"Created at"),(0,u.Z)(r,s.NAME,"Name"),(0,u.Z)(r,s.PIPELINE,"Pipeline"),(0,u.Z)(r,s.STATUS,"Status"),(0,u.Z)(r,s.TYPE,"Type"),r)}}]);