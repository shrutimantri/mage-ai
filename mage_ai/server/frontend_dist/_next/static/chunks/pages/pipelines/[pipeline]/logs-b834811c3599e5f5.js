(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[4794],{74395:function(e,n,t){"use strict";t.d(n,{M:function(){return l},W:function(){return o}});var r=t(38626),i=t(46684),o=34*t(70515).iI,l=r.default.div.withConfig({displayName:"indexstyle__BeforeStyle",componentId:"sc-12ee2ib-0"})(["min-height:calc(100vh - ","px);"],i.Mz)},88785:function(e,n,t){"use strict";t.d(n,{J:function(){return u},U:function(){return c}});var r=t(38626),i=t(44897),o=t(42631),l=t(70515),c=r.default.div.withConfig({displayName:"indexstyle__CardStyle",componentId:"sc-m7tlau-0"})(["border-radius:","px;border-style:solid;border-width:2px;height:","px;margin-right:","px;padding:","px;width:","px;"," ",""],o.TR,14*l.iI,l.cd*l.iI,l.cd*l.iI,40*l.iI,(function(e){return!e.selected&&"\n    border-color: ".concat((e.theme.borders||i.Z.borders).light,";\n  ")}),(function(e){return e.selected&&"\n    border-color: ".concat((e.theme.interactive||i.Z.interactive).linkPrimary,";\n  ")})),u=r.default.div.withConfig({displayName:"indexstyle__DateSelectionContainer",componentId:"sc-m7tlau-1"})(["border-radius:","px;padding:","px;"," "," ",""],o.n_,l.tr,(function(e){return"\n    background-color: ".concat((e.theme.interactive||i.Z.interactive).defaultBackground,";\n  ")}),(function(e){return e.absolute&&"\n    position: absolute;\n    z-index: 2;\n    right: 0;\n    top: ".concat(2.5*l.iI,"px;\n  ")}),(function(e){return e.topPosition&&"\n    top: -".concat(42*l.iI,"px;\n  ")}))},70320:function(e,n,t){"use strict";t.d(n,{h:function(){return l},q:function(){return o}});var r=t(78419),i=t(53808);function o(){return(0,i.U2)(r.am,null)||!1}function l(e){return"undefined"!==typeof e&&(0,i.t8)(r.am,e),e}},14805:function(e,n,t){"use strict";var r=t(82394),i=t(44495),o=t(55485),l=t(44085),c=t(38276),u=t(30160),a=t(88785),s=t(70515),d=t(86735),p=t(28598);function f(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function h(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?f(Object(t),!0).forEach((function(n){(0,r.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):f(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}n.Z=function(e){var n=e.localTime,t=e.selectedDate,r=e.selectedTime,f=e.setSelectedDate,v=e.setSelectedTime,m=e.topPosition;return(0,p.jsxs)(a.J,{absolute:!0,topPosition:m,children:[(0,p.jsx)(i.ZP,{onChange:f,value:t}),(0,p.jsx)(c.Z,{mb:2}),(0,p.jsxs)(o.ZP,{alignItems:"center",children:[(0,p.jsxs)(u.ZP,{default:!0,large:!0,children:["Time (",n?"Local":"UTC","):"]}),(0,p.jsx)(c.Z,{pr:2}),(0,p.jsx)(l.Z,{compact:!0,monospace:!0,onChange:function(e){e.preventDefault(),v((function(n){return h(h({},n),{},{hour:e.target.value})}))},paddingRight:5*s.iI,placeholder:"HH",value:null===r||void 0===r?void 0:r.hour,children:(0,d.m5)(24,0).map((function(e){return String(e).padStart(2,"0")})).map((function(e){return(0,p.jsx)("option",{value:e,children:e},"hour_".concat(e))}))}),(0,p.jsx)(c.Z,{px:1,children:(0,p.jsx)(u.ZP,{bold:!0,large:!0,children:":"})}),(0,p.jsx)(l.Z,{compact:!0,monospace:!0,onChange:function(e){e.preventDefault(),v((function(n){return h(h({},n),{},{minute:e.target.value})}))},paddingRight:5*s.iI,placeholder:"MM",value:null===r||void 0===r?void 0:r.minute,children:(0,d.m5)(60,0).map((function(e){return String(e).padStart(2,"0")})).map((function(e){return(0,p.jsx)("option",{value:e,children:e},"minute_".concat(e))}))})]})]})}},90299:function(e,n,t){"use strict";t.d(n,{Z:function(){return m}});var r=t(82684),i=t(71180),o=t(55485),l=t(64888),c=t(38276),u=t(30160),a=t(8059),s=t(38626),d=t(70515),p=t(47041),f=s.default.div.withConfig({displayName:"indexstyle__TabsContainerStyle",componentId:"sc-segf7l-0"})(["padding-left:","px;padding-right:","px;"," "," ",""],d.cd*d.iI,d.cd*d.iI,(function(e){return e.noPadding&&"\n    padding: 0;\n  "}),(function(e){return e.allowScroll&&"\n    overflow: auto;\n  "}),p.w5),h=t(3314),v=t(28598);var m=function(e){var n=e.allowScroll,t=e.compact,s=e.contained,p=e.noPadding,m=e.onClickTab,g=e.regularSizeText,j=e.selectedTabUUID,b=e.small,x=e.tabs,_=(0,r.useMemo)((function(){var e=x.length,n=[];return x.forEach((function(r,s){var p=r.Icon,f=r.IconSelected,x=r.label,_=r.uuid,y=_===j,Z=y&&f||p,O=x?x():_,P=(0,v.jsxs)(o.ZP,{alignItems:"center",children:[Z&&(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(Z,{default:!y,size:2*d.iI}),(0,v.jsx)(c.Z,{mr:1})]}),(0,v.jsx)(u.ZP,{bold:!0,default:!y,noWrapping:!0,small:!g,children:O})]});s>=1&&e>=2&&n.push((0,v.jsx)("div",{style:{marginLeft:1.5*d.iI}},"spacing-".concat(_))),y?n.push((0,v.jsx)(l.Z,{backgroundGradient:a.yr,backgroundPanel:!0,borderLess:!0,borderWidth:2,compact:t||b,onClick:function(e){(0,h.j)(e),m(r)},paddingUnitsHorizontal:1.75,paddingUnitsVertical:1.25,small:b,children:P},_)):n.push((0,v.jsx)("div",{style:{padding:2},children:(0,v.jsx)(i.Z,{borderLess:!0,compact:t||b,default:!0,onClick:function(e){(0,h.j)(e),m(r)},outline:!0,small:b,children:P})},"button-tab-".concat(_)))})),n}),[t,m,j,b,x]),y=(0,v.jsx)(o.ZP,{alignItems:"center",children:_});return s?y:(0,v.jsx)(f,{allowScroll:n,noPadding:p,children:y})}},59860:function(e,n,t){"use strict";t.r(n),t.d(n,{default:function(){return sn}});var r,i=t(77837),o=t(82394),l=t(38860),c=t.n(l),u=t(38626),a=t(4804),s=t(82684),d=t(44425),p=t(15338),f=t(71180),h=t(70652),v=t(39867),m=t(55485),g=t(38276),j=t(30160),b=t(74395),x=t(70515),_=u.default.div.withConfig({displayName:"indexstyle__FilterRowStyle",componentId:"sc-kvapsi-0"})(["padding-bottom:","px;padding-top:","px;"],x.iI/2,x.iI/2);!function(e){e.CRITICAL="CRITICAL",e.DEBUG="DEBUG",e.ERROR="ERROR",e.EXCEPTION="EXCEPTION",e.INFO="INFO",e.LOG="LOG",e.WARNING="WARNING"}(r||(r={}));var y,Z=[r.CRITICAL,r.DEBUG,r.ERROR,r.EXCEPTION,r.INFO,r.LOG,r.WARNING];!function(e){e.LAST_HOUR="Last hour",e.LAST_DAY="Last day",e.LAST_WEEK="Last week",e.LAST_30_DAYS="Last 30 days",e.CUSTOM_RANGE="Custom range"}(y||(y={}));var O=t(42631),P=t(79633);var k,I=(0,u.css)(["",""],(function(e){return"\n    background-color: ".concat(function(e){var n=e.critical,t=e.debug,r=e.error,i=e.exception,o=e.info,l=e.log,c=e.warning;return n?P.Zl:t?P.EG:r?P.hl:i?P.hM:o?P.gN:l?P.Wd:c?P.$R:"transparent"}(e),";\n  ")})),w=u.default.div.withConfig({displayName:"indexstyle__LogLevelIndicatorStyle",componentId:"sc-1e2zh7m-0"})([""," border-radius:","px;height:12px;width:5px;"],I,O.n_),S=t(81728),C=t(55283),L=t(15610),E=t(28598);function D(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function T(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?D(Object(t),!0).forEach((function(n){(0,o.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):D(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}!function(e){e.BLOCK_RUN_ID="block_run_id[]",e.BLOCK_TYPE="block_type[]",e.BLOCK_UUID="block_uuid[]",e.LEVEL="level[]",e.PIPELINE_RUN_ID="pipeline_run_id[]",e.PIPELINE_SCHEDULE_ID="pipeline_schedule_id[]"}(k||(k={}));var A=function(e){var n=e.blocks,t=e.query,r=(0,s.useContext)(u.ThemeContext),i=(0,s.useMemo)((function(){return t[k.LEVEL]}),[t]),l=(0,s.useMemo)((function(){return t[k.BLOCK_TYPE]}),[t]),c=(0,s.useMemo)((function(){return t[k.BLOCK_UUID]}),[t]),a=(0,s.useMemo)((function(){return t[k.PIPELINE_SCHEDULE_ID]}),[t]),p=(0,s.useMemo)((function(){return t[k.PIPELINE_RUN_ID]}),[t]),y=(0,s.useMemo)((function(){return t[k.BLOCK_RUN_ID]}),[t]);return(0,E.jsx)(b.M,{children:(0,E.jsxs)(g.Z,{p:x.cd,children:[(0,E.jsxs)(g.Z,{mb:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,default:!0,large:!0,children:"Log level"})}),Z.map((function(e){return(0,E.jsx)(f.Z,{noBackground:!0,noBorder:!0,noPadding:!0,onClick:function(){return(0,L.g_)(t,{level:e},{isList:!0})},children:(0,E.jsx)(_,{children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(h.Z,{checked:Array.isArray(i)&&(null===i||void 0===i?void 0:i.includes(String(e)))}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(w,T({},(0,o.Z)({},e.toLowerCase(),!0))),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(j.ZP,{disableWordBreak:!0,children:(0,S.kC)(e.toLowerCase())})]})})},e)}))]}),(0,E.jsxs)(g.Z,{mb:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,default:!0,large:!0,children:"Block type"})}),[d.tf.DATA_LOADER,d.tf.TRANSFORMER,d.tf.DATA_EXPORTER].map((function(e){return(0,E.jsx)(f.Z,{noBackground:!0,noBorder:!0,noPadding:!0,onClick:function(){return(0,L.g_)(t,{block_type:e},{isList:!0})},children:(0,E.jsx)(_,{children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(h.Z,{checked:Array.isArray(l)&&(null===l||void 0===l?void 0:l.includes(String(e)))}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(v.Z,{color:(0,C.qn)(e,{theme:r}).accent,size:1.5*x.iI,square:!0}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(j.ZP,{disableWordBreak:!0,muted:!0,monospace:!0,children:e})]})})},e)}))]}),(0,E.jsxs)(g.Z,{mb:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,default:!0,large:!0,children:"Block"})}),n.filter((function(e){var n=e.type;return d.tf.SCRATCHPAD!==n})).map((function(e){return(0,E.jsx)(f.Z,{noBackground:!0,noBorder:!0,noPadding:!0,onClick:function(){return(0,L.g_)(t,{block_uuid:e.uuid},{isList:!0,resetLimitParams:!0})},children:(0,E.jsx)(_,{children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(h.Z,{checked:Array.isArray(c)&&(null===c||void 0===c?void 0:c.includes(String(e.uuid)))}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(v.Z,{color:(0,C.qn)(e.type,{blockColor:e.color,theme:r}).accent,size:1.5*x.iI,square:!0}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(j.ZP,{disableWordBreak:!0,monospace:!0,muted:!0,children:e.uuid})]})})},e.uuid)}))]}),(null===a||void 0===a?void 0:a.length)&&(0,E.jsxs)(g.Z,{mb:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,default:!0,large:!0,children:"Trigger"})}),a.map((function(e){return(0,E.jsx)(f.Z,{noBackground:!0,noBorder:!0,noPadding:!0,onClick:function(){return(0,L.g_)(t,{pipeline_schedule_id:e},{isList:!0})},children:(0,E.jsx)(_,{children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(h.Z,{checked:Array.isArray(a)&&(null===a||void 0===a?void 0:a.includes(e))}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(j.ZP,{disableWordBreak:!0,monospace:!0,children:e})]})})},"pipeline-schedule-".concat(e))}))]}),(null===p||void 0===p?void 0:p.length)&&(0,E.jsxs)(g.Z,{mb:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,default:!0,large:!0,children:"Pipeline run"})}),p.map((function(e){return(0,E.jsx)(f.Z,{noBackground:!0,noBorder:!0,noPadding:!0,onClick:function(){return(0,L.g_)(t,{pipeline_run_id:e},{isList:!0})},children:(0,E.jsx)(_,{children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(h.Z,{checked:Array.isArray(p)&&(null===p||void 0===p?void 0:p.includes(e))}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(j.ZP,{disableWordBreak:!0,monospace:!0,children:e})]})})},"pipeline-run-".concat(e))}))]}),(null===y||void 0===y?void 0:y.length)&&(0,E.jsxs)(g.Z,{mb:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,default:!0,large:!0,children:"Block run"})}),y.map((function(e){return(0,E.jsx)(f.Z,{noBackground:!0,noBorder:!0,noPadding:!0,onClick:function(){return(0,L.g_)(t,{block_run_id:e},{isList:!0})},children:(0,E.jsx)(_,{children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(h.Z,{checked:Array.isArray(y)&&(null===y||void 0===y?void 0:y.includes(e))}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(j.ZP,{disableWordBreak:!0,monospace:!0,children:e})]})})},"block-run-".concat(e))}))]})]})})},R=t(97618),N=t(93369),B=t(75582),M=t(90299),U=t(48670),W=t(75499),J=u.default.div.withConfig({displayName:"indexstyle__BarStyle",componentId:"sc-1r43sbu-0"})([""," height:","px;width:100%;"],I,.5*x.iI),Y=u.default.div.withConfig({displayName:"indexstyle__BadgeStyle",componentId:"sc-1r43sbu-1"})([""," border-radius:","px;display:inline-block;padding:","px ","px;"],I,O.BG,.25*x.iI,.5*x.iI),H=t(72473),z=t(92083),G=t.n(z);function q(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function K(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?q(Object(t),!0).forEach((function(n){(0,o.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):q(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var F=/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$/,V=/([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}) (.+)/;function X(e){var n=e.content.trim().split(V),t=[],r=[];return n.forEach((function(e){var n=e.trim();F.test(e)?(r.length>=1&&t.push(r.join(" ").trim()),r=[e]):r.filter((function(e){return e})).length<=1&&n&&r.push(n)})),t.push(r.join(" ").trim()),t.map((function(n){return function(e){var n=e.content.trim().match(V),t=null===n||void 0===n?void 0:n[1],r=null===n||void 0===n?void 0:n[2],i={};return r&&(0,S.Pb)(r)&&(i=JSON.parse(r)),K(K({},e),{},{createdAt:t,data:i})}(K(K({},e),{},{content:n}))}))}function Q(e,n){return e?null!==n&&void 0!==n&&n.localTimezone?G().unix(e).local().format():G().unix(e).utc().format("YYYY-MM-DD HH:mm:ss.SSS"):""}var $=t(86735);function ee(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function ne(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?ee(Object(t),!0).forEach((function(n){(0,o.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):ee(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var te=["error","error_stack","error_stacktrace"],re={uuid:"Details"},ie={uuid:"Errors"};var oe=function(e){var n=e.log,t=e.onClose,r=e.selectedTab,i=e.setSelectedTab,l=(0,s.useState)(!1),c=l[0],u=l[1],a=n.data,d=n.name,h=n.path,v=a||{},b=v.error,_=v.error_stack,y=v.error_stacktrace,Z=v.level,O=v.timestamp,P=(0,o.Z)({},Z.toLowerCase(),!0),k=(0,s.useMemo)((function(){var e=[["file name",d],["file path",h]];return Object.entries(a).forEach((function(n){var t=(0,B.Z)(n,2),r=t[0],i=t[1];te.includes(r)||e.push([r,i])})),y&&e.push(["error",y]),(0,$.YC)(e,(function(e){var n=(0,B.Z)(e,2),t=n[0];n[1];return t}))}),[a,y,d,h]),I=(0,s.useMemo)((function(){var e=[re];return b&&e.push(ie),(0,E.jsx)(M.Z,{onClickTab:i,selectedTabUUID:null===r||void 0===r?void 0:r.uuid,tabs:e})}),[b,r,i]);return(0,E.jsxs)("div",{children:[(0,E.jsx)(J,ne({},P)),(0,E.jsx)(g.Z,{p:x.cd,children:(0,E.jsxs)(m.ZP,{alignItems:"center",justifyContent:"space-between",children:[(0,E.jsxs)(R.Z,{alignItems:"center",children:[(0,E.jsx)(Y,ne(ne({},P),{},{children:(0,E.jsx)(j.ZP,{bold:!0,inverted:!0,monospace:!0,small:!0,children:Z})})),(0,E.jsx)(g.Z,{mr:x.cd}),(0,E.jsx)(j.ZP,{monospace:!0,children:Q(O)})]}),(0,E.jsx)(f.Z,{iconOnly:!0,noBackground:!0,onClick:function(){return t()},children:(0,E.jsx)(H.x8,{size:1.5*x.iI})})]})}),(0,E.jsx)(p.Z,{medium:!0}),(0,E.jsx)(g.Z,{py:x.cd,children:I}),re.uuid===(null===r||void 0===r?void 0:r.uuid)&&(0,E.jsx)(W.Z,{columnFlex:[null,1],columnMaxWidth:function(e){return 1===e?"100px":null},rows:null===k||void 0===k?void 0:k.map((function(e,n){var t=(0,B.Z)(e,2),r=t[0],i=t[1],o="message"===r,l="tags"===r,a=i,s=i;return l?s=a=(0,S.Pb)(i)?JSON.parse(JSON.stringify(i,null,2)):JSON.stringify(i,null,2):o&&c&&(0,S.Pb)(i)&&(s=JSON.stringify(JSON.parse(i),null,2),a=(0,E.jsx)("pre",{children:s})),"object"===typeof a&&(a=JSON.stringify(a,null,2),a=(0,E.jsx)("pre",{children:a})),"object"===typeof s&&(s=JSON.stringify(s)),[(0,E.jsx)(j.ZP,{monospace:!0,muted:!0,children:r},"".concat(r,"_").concat(n,"_key")),(0,E.jsxs)(E.Fragment,{children:[(0,E.jsxs)(j.ZP,{monospace:!0,textOverflow:!0,title:s,whiteSpaceNormal:o&&c||l,wordBreak:o&&c||l,children:[!l&&a,l&&(0,E.jsx)("pre",{children:a})]},"".concat(r,"_").concat(n,"_val")),o&&(0,E.jsx)(U.Z,{muted:!0,onClick:function(){return u((function(e){return!e}))},children:c?"Click to hide log":"Click to show full log message"})]})]})),uuid:"LogDetail"}),ie.uuid===(null===r||void 0===r?void 0:r.uuid)&&(0,E.jsxs)(g.Z,{mb:5,px:x.cd,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,children:"Error"})}),null===b||void 0===b?void 0:b.map((function(e){return e.split("\n").map((function(e){return e.split("\\n").map((function(e){return(0,E.jsx)(j.ZP,{default:!0,monospace:!0,preWrap:!0,small:!0,children:e},e)}))}))})),_&&(0,E.jsxs)(g.Z,{mt:3,children:[(0,E.jsx)(g.Z,{mb:1,children:(0,E.jsx)(j.ZP,{bold:!0,children:"Stack trace"})}),null===_||void 0===_?void 0:_.map((function(e){return null===e||void 0===e?void 0:e.map((function(e){return(0,E.jsx)(j.ZP,{default:!0,monospace:!0,preWrap:!0,small:!0,children:e},e)}))}))]})]})]})},le=t(21831),ce=t(89565),ue=t.n(ce),ae=t(12691),se=t.n(ae),de=t(78141),pe=t(57653),fe=t(98464),he=t(46684),ve=t(44897),me=t(47041),ge=u.default.div.withConfig({displayName:"indexstyle__TableContainer",componentId:"sc-16j4vp6-0"})([".table_row > div{margin:","px ","px;}div{","}"],.5*x.iI,x.iI,me.w5),je=u.default.div.withConfig({displayName:"indexstyle__TableHeadStyle",componentId:"sc-16j4vp6-1"})(["display:flex;align-items:center;overflow:hidden;",""],(function(e){return"\n    border-bottom: 1px solid ".concat((e.theme||ve.Z).borders.medium2,";\n  ")})),be=u.default.div.withConfig({displayName:"indexstyle__TableRowStyle",componentId:"sc-16j4vp6-2"})(["display:flex;align-items:center;"," "," ",""],(function(e){return"\n    border-bottom: 1px solid ".concat((e.theme||ve.Z).borders.light,";\n\n    &:hover {\n      cursor: pointer;\n    }\n  ")}),(function(e){return!e.selected&&"\n    &:hover {\n      background: ".concat((e.theme.interactive||ve.Z.interactive).rowHoverBackground,";\n      cursor: pointer;\n    }\n  ")}),(function(e){return e.selected&&"\n    background-color: ".concat((e.theme.interactive||ve.Z.interactive).activeBorder,";\n  ")})),xe=t(93859),_e=t(53808),ye=t(69419),Ze=function(e){var n=(0,ye.iV)(),t=((null===n||void 0===n?void 0:n[k.PIPELINE_SCHEDULE_ID])||[]).join(",");return"".concat(e,"/logs/triggers/").concat(t)},Oe=t(70320),Pe=t(19183);function ke(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function Ie(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?ke(Object(t),!0).forEach((function(n){(0,o.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):ke(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var we="log_uuid";var Se,Ce,Le,Ee=function(e){var n=e.autoScrollLogs,t=e.blocksByUUID,r=e.tableInnerRef,i=e.logs,l=e.onRowClick,c=e.pipeline,u=e.query,a=e.saveScrollPosition,d=e.setSelectedLog,p=e.themeContext,f=(0,Oe.q)(),h=(0,Pe.i)().height,b=(0,s.useRef)(null),_=(0,s.useMemo)((function(){return pe.qL.INTEGRATION===(null===c||void 0===c?void 0:c.type)}),[c.type]),y=(0,fe.Z)(i);(0,s.useEffect)((function(){var e;n&&(y||[]).length!==(i||[]).length&&(null===r||void 0===r||null===(e=r.current)||void 0===e||e.scrollIntoView(!1))}),[n,i,y,r]);var Z=(0,s.useMemo)((function(){return Ze(null===c||void 0===c?void 0:c.uuid)}),[null===c||void 0===c?void 0:c.uuid]);(0,s.useEffect)((function(){var e;a&&(null===b||void 0===b||null===(e=b.current)||void 0===e||e.scrollTo((0,_e.U2)(Z,0)))}),[a,Z]);var O=Object.keys(t||{});if(_){var P,k,I=((null===c||void 0===c||null===(P=c.data_integration)||void 0===P||null===(k=P.catalog)||void 0===k?void 0:k.streams)||[]).map((function(e){return e.tap_stream_id})),S=new Set;O.forEach((function(e){I.forEach((function(n){return S.add("".concat(e,":").concat(n,":0"))}))})),O=Array.from(S)}var D=Math.max.apply(Math,(0,le.Z)(O.map((function(e){return e.length})))),T=Math.min(D*xe.r+12+8,50*x.iI),A=[{uuid:"_",width:28},{uuid:"Date",width:f?202:214},{uuid:"Block",width:T+16},{uuid:"Message"},{uuid:"_"}],N=(0,s.useCallback)((function(e){var n,t=e.data,r=e.index,i=e.style,c=t.blocksByUUID,a=t.logs,s=t.themeContext,p=a[r],h=p.content,b=p.data,y=p.name,Z=b||{},O=Z.block_uuid,P=Z.level,k=Z.message,I=Z.pipeline_uuid,S=Z.timestamp,D=Z.uuid,A=k||h;Array.isArray(A)?A=A.join(" "):"object"===typeof A&&(A=JSON.stringify(A));var N,B,M=O||y.split(".log")[0],W=M.split(":");_&&(M=W[0],N=W[1],B=W[2]);var J=c[M];if(J||(J=c[W[0]]),J){var Y=(0,C.qn)(J.type,{blockColor:J.color,theme:s}).accent;n=(0,E.jsx)(m.ZP,{alignItems:"center",children:(0,E.jsx)(se(),{as:"/pipelines/".concat(I,"/edit?block_uuid=").concat(M),href:"/pipelines/[pipeline]/edit",passHref:!0,children:(0,E.jsxs)(U.Z,{block:!0,fullWidth:!0,sameColorAsText:!0,verticalAlignContent:!0,children:[(0,E.jsx)(v.Z,{color:Y,size:1.5*x.iI,square:!0}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsxs)(j.ZP,{disableWordBreak:!0,monospace:!0,noWrapping:!0,title:O,width:T-16,children:[M,N&&":",N&&(0,E.jsx)(j.ZP,{default:!0,inline:!0,monospace:!0,children:N}),B>=0&&":",B>=0&&(0,E.jsx)(j.ZP,{default:!0,inline:!0,monospace:!0,children:B})]})]})})})}return(0,E.jsxs)(be,{className:"table_row",onClick:function(){var e,n,t=a[r],i=null===(e=t.data)||void 0===e?void 0:e.uuid;u.log_uuid===i&&(i=null),null!==(n=t.data)&&void 0!==n&&n.error?null===l||void 0===l||l(ie):null===l||void 0===l||l(re),(0,L.u7)((0,o.Z)({},we,i)),d(i?t:null)},selected:(null===u||void 0===u?void 0:u.log_uuid)&&(null===u||void 0===u?void 0:u.log_uuid)===D,style:Ie({},i),children:[(0,E.jsx)(R.Z,{alignItems:"center",justifyContent:"center",children:(0,E.jsx)(w,Ie({},(0,o.Z)({},null===P||void 0===P?void 0:P.toLowerCase(),!0)))},"log_type"),(0,E.jsx)(R.Z,{children:(0,E.jsx)(j.ZP,{default:!0,monospace:!0,noWrapping:!0,small:f,children:Q(S,{localTimezone:f})},"log_timestamp")}),(0,E.jsx)(R.Z,{style:{minWidth:T,width:T},children:n}),(0,E.jsx)(R.Z,{style:{overflow:"auto"},children:(0,E.jsx)(j.ZP,{monospace:!0,textOverflow:!0,title:A,children:(0,E.jsx)(ue(),{children:A})},"log_message")}),(0,E.jsx)(R.Z,{flex:"1",justifyContent:"flex-end",children:(0,E.jsx)(H._Q,{default:!0,size:2*x.iI})},"chevron_right_icon")]})}),[T,f,_,l,u,d]);return(0,E.jsxs)(ge,{children:[(0,E.jsx)(je,{children:A.map((function(e,n){return(0,E.jsx)(R.Z,{alignItems:"center",style:{height:4*x.iI,minWidth:e.width||null,width:e.width||null},children:"_"!==e.uuid&&(0,E.jsx)(j.ZP,{bold:!0,leftAligned:!0,monospace:!0,muted:!0,children:e.uuid})},"".concat(e,"_").concat(n))}))}),(0,E.jsx)(de.t7,{height:h-he.Mz-86-34-58,innerRef:r,itemCount:i.length,itemData:{blocksByUUID:t,logs:i,pipeline:c,themeContext:p},itemSize:3.75*x.iI,onScroll:function(e){var n=e.scrollOffset,t=e.scrollDirection;!a||"forward"===t&&0===n||(0,_e.t8)(Z,n)},ref:b,width:"100%",children:N})]})},De=t(34376),Te=t(14805),Ae=t(50724),Re=t(44085),Ne=t(17488),Be="_limit",Me="_offset",Ue=20,We=[y.LAST_HOUR,y.LAST_DAY,y.LAST_WEEK,y.LAST_30_DAYS],Je=(Se={},(0,o.Z)(Se,y.LAST_HOUR,3600),(0,o.Z)(Se,y.LAST_DAY,86400),(0,o.Z)(Se,y.LAST_WEEK,604800),(0,o.Z)(Se,y.LAST_30_DAYS,2592e3),Se),Ye=t(48277),He=t(3917),ze=t(42122);function Ge(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function qe(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?Ge(Object(t),!0).forEach((function(n){(0,o.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):Ge(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}!function(e){e.START="start_timestamp",e.END="end_timestamp"}(Le||(Le={}));var Ke=(Ce={},(0,o.Z)(Ce,Be,Ue),(0,o.Z)(Ce,Me,0),Ce),Fe={blackBorder:!0,inline:!0,paddingBottom:.75*x.iI,paddingTop:.75*x.iI};var Ve=function(e){var n=e.allPastLogsLoaded,t=e.loadNewerLogInterval,r=e.loadPastLogInterval,i=e.saveScrollPosition,l=e.selectedRange,c=e.setSelectedRange,u=(0,s.useState)(null),a=u[0],d=u[1],p=(0,s.useState)(null),h=p[0],v=p[1],b=(0,s.useState)({hour:"00",minute:"00"}),_=b[0],Z=b[1],O=(0,s.useState)(new Date),P=O[0],k=O[1],I=(0,s.useState)({hour:(0,He.lJ)(String((new Date).getUTCHours())),minute:(0,He.lJ)(String((new Date).getUTCMinutes()))}),w=I[0],S=I[1],C=(0,De.useRouter)().query.pipeline,D=(0,ye.iV)(),T=(0,fe.Z)(D);(0,s.useEffect)((function(){if(!(0,ze.Xy)(D,T)){var e=D.start_timestamp,n=D.end_timestamp;if(e){var t=(0,He.Pc)(e),r=t.date,i=t.hour,o=t.minute;v(r),Z({hour:(0,He.lJ)(i),minute:(0,He.lJ)(o)});var l=Math.ceil(Date.now()/1e3)-e;Math.abs(l-Je[y.LAST_DAY])<=60&&c(y.LAST_DAY)}if(n){var u=(0,He.Pc)(n),a=u.date,s=u.hour,d=u.minute;k(a),S({hour:(0,He.lJ)(s),minute:(0,He.lJ)(d)})}}}),[D,T]);var A=(0,s.useCallback)((function(){if(i){var e=Ze(C);(0,_e.t8)(e,0)}}),[C,i]);return(0,E.jsx)(g.Z,{py:1,children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(N.ZP,qe(qe({},Fe),{},{disabled:n,onClick:function(){A(),r()},uuid:"logs/load_older_logs",children:n?"All past logs within range loaded":"Load older logs"})),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(N.ZP,qe(qe({},Fe),{},{disabled:(null===D||void 0===D?void 0:D._offset)<=0,onClick:function(){A(),t()},uuid:"logs/load_newer_logs",children:"Load newer logs"})),(0,E.jsx)(g.Z,{mr:2}),(0,E.jsx)(Re.Z,{compact:!0,defaultColor:!0,onChange:function(e){e.preventDefault();var n=e.target.value;if(A(),c(n),We.includes(n)){var t,r=(0,Ye.JI)(Je[n]);(0,L.u7)(qe((t={},(0,o.Z)(t,Le.START,r),(0,o.Z)(t,Le.END,null),t),Ke))}},paddingRight:4*x.iI,placeholder:"Select time range",value:l,children:Object.values(y).map((function(e){return(0,E.jsx)("option",{value:e,children:e},e)}))}),(0,E.jsx)(g.Z,{mr:1}),l===y.CUSTOM_RANGE&&(0,E.jsxs)(E.Fragment,{children:[(0,E.jsx)(Ne.Z,{compact:!0,defaultColor:!0,onClick:function(){return d(0)},paddingRight:0,placeholder:"Start",value:h?(0,He.AY)(h,null===_||void 0===_?void 0:_.hour,null===_||void 0===_?void 0:_.minute):""}),(0,E.jsx)(Ae.Z,{onClickOutside:function(){return d(null)},open:0===a,style:{position:"relative"},children:(0,E.jsx)(Te.Z,{selectedDate:h,selectedTime:_,setSelectedDate:v,setSelectedTime:Z})}),(0,E.jsx)(g.Z,{px:1,children:(0,E.jsx)(j.ZP,{children:"to"})}),(0,E.jsx)(Ne.Z,{compact:!0,defaultColor:!0,onClick:function(){return d(1)},paddingRight:0,placeholder:"End",value:P?(0,He.AY)(P,null===w||void 0===w?void 0:w.hour,null===w||void 0===w?void 0:w.minute):""}),(0,E.jsx)(Ae.Z,{onClickOutside:function(){return d(null)},open:1===a,style:{position:"relative"},children:(0,E.jsx)(Te.Z,{selectedDate:P,selectedTime:w,setSelectedDate:k,setSelectedTime:S})}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(f.Z,{borderRadius:x.iI/2,onClick:function(){var e;A();var n=(0,He.BP)(h,_.hour,_.minute),t=(0,He.BP)(P,w.hour,w.minute);(0,L.u7)(qe((e={},(0,o.Z)(e,Le.START,(0,He.A5)(n)),(0,o.Z)(e,Le.END,(0,He.A5)(t)),e),Ke))},padding:"".concat(x.iI/2,"px"),primary:!0,children:"Search"})]})]})})},Xe=t(75457),Qe=t(93808),$e=t(4190),en=t(69650),nn=t(35686),tn=t(78419),rn=t(28795);function on(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function ln(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?on(Object(t),!0).forEach((function(n){(0,o.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):on(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var cn="pipeline_run_id[]",un="block_run_id[]";function an(e){var n=e.pipeline,t=(0,s.useContext)(u.ThemeContext),r=(0,s.useRef)(null),i=n.uuid,l=(0,s.useState)(null),c=l[0],f=l[1],h=(0,s.useState)(null),v=h[0],b=h[1],_=(0,s.useState)(null),Z=_[0],O=_[1],P=(0,s.useState)(null),I=P[0],w=P[1],C=(0,s.useState)(re),D=C[0],T=C[1],B=(0,s.useState)((0,_e.U2)(tn.Tz,!0)),M=B[0],U=B[1],W=nn.ZP.pipelines.detail(i,{includes_content:!1,includes_outputs:!1},{revalidateOnFocus:!1}).data,J=(0,s.useMemo)((function(){return ln(ln({},null===W||void 0===W?void 0:W.pipeline),{},{uuid:i})}),[W,i]),Y=(null===J||void 0===J?void 0:J.type)===pe.qL.INTEGRATION,H=(0,s.useMemo)((function(){return J.blocks||[]}),[J]),z=(0,s.useMemo)((function(){var e=(0,$.HK)(H,(function(e){return e.uuid}));if(Y){var n,t=(0,$.sE)(H,(function(e){var n=e.type;return d.tf.DATA_LOADER===n})),r=t?(0,a.Qc)(t.content):{},i=((null===r||void 0===r||null===(n=r.catalog)||void 0===n?void 0:n.streams)||[]).reduce((function(e,n){var t=n.tap_stream_id,r={};return H.forEach((function(e){var n=e.uuid,i=e.type,o="".concat(n,":").concat(t);r[o]={type:i}})),ln(ln({},e),r)}),{});e=ln(ln({},i),e)}return e}),[H,Y]),G=(0,ye.iV)(),q=(0,s.useMemo)((function(){return(null===G||void 0===G?void 0:G.hasOwnProperty(k.PIPELINE_SCHEDULE_ID))&&!(null!==G&&void 0!==G&&G.hasOwnProperty(k.LEVEL))&&!(null!==G&&void 0!==G&&G.hasOwnProperty(k.BLOCK_TYPE))&&!(null!==G&&void 0!==G&&G.hasOwnProperty(k.BLOCK_UUID))}),[G]),K=!(null!==G&&void 0!==G&&G.start_timestamp)&&!(null!==G&&void 0!==G&&G.hasOwnProperty(cn)||null!==G&&void 0!==G&&G.hasOwnProperty(un)),F=(0,Ye.JI)(Je[y.LAST_DAY]),V=nn.ZP.logs.pipelines.list(c?i:null,(0,ze.gR)(K?ln(ln({},c),{},{start_timestamp:F}):c,[we]),{refreshInterval:5e3}),Q=V.data,ee=V.mutate,ne=!Q,te=(0,s.useMemo)((function(){var e;if(null!==Q&&void 0!==Q&&null!==(e=Q.logs)&&void 0!==e&&e[0]){var n,t=(null===(n=Q.logs)||void 0===n?void 0:n[0])||{};return{blockRunLogs:t.block_run_logs,pipelineRunLogs:t.pipeline_run_logs,totalBlockRunLogCount:t.total_block_run_log_count,totalPipelineRunLogCount:t.total_pipeline_run_log_count}}return{blockRunLogs:[],pipelineRunLogs:[],totalBlockRunLogCount:0,totalPipelineRunLogCount:0}}),[Q]),ie=te.blockRunLogs,le=te.pipelineRunLogs,ce=te.totalBlockRunLogCount,ue=te.totalPipelineRunLogCount,ae=+(null===G||void 0===G?void 0:G._limit)>=ce&&+(null===G||void 0===G?void 0:G._limit)>=ue,se=(0,s.useMemo)((function(){return(0,$.YC)(ie.concat(le).reduce((function(e,n){return e.concat(X(n))}),[]),(function(e){var n=e.data;return(null===n||void 0===n?void 0:n.timestamp)||0}))}),[ie,le]),de=(0,s.useMemo)((function(){return se.filter((function(e){var n=e.data,t=[];if(!c)return!0;if(t.push(!(0,ze.Qr)(n)),c["level[]"]&&t.push(c["level[]"].includes(null===n||void 0===n?void 0:n.level)),c["block_type[]"]){var r,i,o=null===n||void 0===n?void 0:n.block_uuid;if(Y)o=null===n||void 0===n||null===(i=n.block_uuid)||void 0===i?void 0:i.split(":").slice(0,2).join(":");t.push(c["block_type[]"].includes(null===(r=z[o])||void 0===r?void 0:r.type))}if(c["pipeline_run_id[]"]){var l=null===n||void 0===n?void 0:n.pipeline_run_id;t.push(c["pipeline_run_id[]"].includes(String(l)))}if(c["block_run_id[]"]){var u=null===n||void 0===n?void 0:n.block_run_id;t.push(c["block_run_id[]"].includes(String(u)))}return t.every((function(e){return e}))}))}),[z,Y,se,c]),he=de.length,me=(0,fe.Z)(G);(0,s.useEffect)((function(){var e;K&&(0,L.u7)((e={},(0,o.Z)(e,Be,Ue),(0,o.Z)(e,Me,0),(0,o.Z)(e,"start_timestamp",F),e))}),[K]),(0,s.useEffect)((function(){(0,ze.Xy)(G,me)||f(G)}),[G,me]);var ge=(0,fe.Z)(v);(0,s.useEffect)((function(){var e=G.log_uuid;!e||v||ge||b(se.find((function(n){var t=n.data;return(null===t||void 0===t?void 0:t.uuid)===e})))}),[se,G,v,ge]);var je=G._limit,be=G._offset,xe=+(je||0),Ze=+(be||0),Oe=Math.max(ce,ue),Pe=(0,s.useCallback)((function(){var e,n=xe,t=Ze;(ce>xe||ue>xe)&&(n=Math.min(Oe,xe+Ue),t=Math.min(Ze+Ue,Oe-Oe%Ue),(0,L.u7)(ln(ln({},G),{},(e={},(0,o.Z)(e,Be,n),(0,o.Z)(e,Me,t),e))))}),[Oe,xe,Ze,G,ce,ue]),ke=(0,s.useCallback)((function(){var e,n=xe,t=Ze;xe>=Ue&&(n=Math.max(Ue,xe-Ue),xe>=Oe&&Oe%Ue!==0&&(n=Oe-Oe%Ue),t=Math.max(0,Ze-Ue),(0,L.u7)(ln(ln({},G),{},(e={},(0,o.Z)(e,Be,n),(0,o.Z)(e,Me,t),e))))}),[Oe,xe,Ze,G]),Ie=(0,s.useCallback)((function(){var e=!M;U(e),(0,_e.t8)(tn.Tz,e)}),[M]),Se=(0,s.useMemo)((function(){return(0,E.jsx)(Ee,{autoScrollLogs:M,blocksByUUID:z,logs:de,onRowClick:T,pipeline:J,query:c,saveScrollPosition:q,setSelectedLog:b,tableInnerRef:r,themeContext:t})}),[M,z,de,J,c,q,t]);return(0,E.jsxs)(Xe.Z,{after:v&&(0,E.jsx)(oe,{log:v,onClose:function(){(0,L.u7)((0,o.Z)({},we,null)),b(null)},selectedTab:D,setSelectedTab:T}),afterHidden:!v,afterWidth:80*x.iI,before:(0,E.jsx)(A,{blocks:H,query:c}),beforeWidth:20*x.iI,breadcrumbs:[{label:function(){return"Logs"}}],errors:I,pageName:rn.M.PIPELINE_LOGS,pipeline:J,setErrors:w,subheader:null,title:function(e){var n=e.name;return"".concat(n," logs")},uuid:"pipeline/logs",children:[(0,E.jsx)(g.Z,{px:x.cd,py:1,children:(0,E.jsxs)(j.ZP,{children:[!ne&&(0,E.jsxs)(E.Fragment,{children:[(0,S.x6)(he)," logs found",(0,E.jsx)(Ve,{allPastLogsLoaded:ae,loadNewerLogInterval:ke,loadPastLogInterval:Pe,saveScrollPosition:q,selectedRange:Z,setSelectedRange:O})]}),ne&&"Searching..."]})}),(0,E.jsx)(p.Z,{light:!0}),ne&&(0,E.jsx)(g.Z,{p:x.cd,children:(0,E.jsx)($e.Z,{})}),!ne&&de.length>=1&&Se,(0,E.jsx)(g.Z,{p:"".concat(1.5*x.iI,"px"),children:(0,E.jsxs)(m.ZP,{alignItems:"center",children:[(0,E.jsx)(N.ZP,ln(ln({},Fe),{},{onClick:function(){"0"===(null===G||void 0===G?void 0:G._offset)&&(null===G||void 0===G?void 0:G._limit)===String(Ue)?ee(null):(0,L.u7)({_limit:Ue,_offset:0})},uuid:"logs/toolbar/load_newest",children:"Load latest logs"})),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(N.ZP,ln(ln({},Fe),{},{backgroundColor:ve.Z.background.page,onClick:function(){var e;null===r||void 0===r||null===(e=r.current)||void 0===e||e.scrollIntoView({behavior:"smooth",block:"end",inline:"nearest"})},uuid:"logs/toolbar/scroll_to_bottomt",children:"Scroll to bottom"})),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsxs)(R.Z,{children:[(0,E.jsx)(j.ZP,{noWrapping:!0,children:"Auto-scroll to new logs"}),(0,E.jsx)(g.Z,{mr:1}),(0,E.jsx)(en.Z,{checked:M,compact:!0,onCheck:Ie})]})]})})]})}an.getInitialProps=function(){var e=(0,i.Z)(c().mark((function e(n){var t;return c().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=n.query.pipeline,e.abrupt("return",{pipeline:{uuid:t}});case 2:case"end":return e.stop()}}),e)})));return function(n){return e.apply(this,arguments)}}();var sn=(0,Qe.Z)(an)},48277:function(e,n,t){"use strict";t.d(n,{JI:function(){return o},uf:function(){return i}});var r=t(75582),i=function(e){var n=String(e).split("."),t=(0,r.Z)(n,2),i=t[0],o=t[1];return"".concat(i.replace(/\B(?=(\d{3})+(?!\d))/g,",")).concat(o?".".concat(o):"")};function o(e){var n=Math.floor(Date.now()/1e3);return e>0?n-e:n}},62453:function(e,n,t){(window.__NEXT_P=window.__NEXT_P||[]).push(["/pipelines/[pipeline]/logs",function(){return t(59860)}])}},function(e){e.O(0,[844,9902,426,4913,4495,6358,9696,8264,5499,1845,5457,3859,9774,2888,179],(function(){return n=62453,e(e.s=n);var n}));var n=e.O();_N_E=n}]);