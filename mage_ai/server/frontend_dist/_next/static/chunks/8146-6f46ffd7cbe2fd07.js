(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[8146],{17717:function(e){!function(){"use strict";var n={114:function(e){function n(e){if("string"!==typeof e)throw new TypeError("Path must be a string. Received "+JSON.stringify(e))}function t(e,n){for(var t,r="",i=0,o=-1,l=0,c=0;c<=e.length;++c){if(c<e.length)t=e.charCodeAt(c);else{if(47===t)break;t=47}if(47===t){if(o===c-1||1===l);else if(o!==c-1&&2===l){if(r.length<2||2!==i||46!==r.charCodeAt(r.length-1)||46!==r.charCodeAt(r.length-2))if(r.length>2){var s=r.lastIndexOf("/");if(s!==r.length-1){-1===s?(r="",i=0):i=(r=r.slice(0,s)).length-1-r.lastIndexOf("/"),o=c,l=0;continue}}else if(2===r.length||1===r.length){r="",i=0,o=c,l=0;continue}n&&(r.length>0?r+="/..":r="..",i=2)}else r.length>0?r+="/"+e.slice(o+1,c):r=e.slice(o+1,c),i=c-o-1;o=c,l=0}else 46===t&&-1!==l?++l:l=-1}return r}var r={resolve:function(){for(var e,r="",i=!1,o=arguments.length-1;o>=-1&&!i;o--){var l;o>=0?l=arguments[o]:(void 0===e&&(e=""),l=e),n(l),0!==l.length&&(r=l+"/"+r,i=47===l.charCodeAt(0))}return r=t(r,!i),i?r.length>0?"/"+r:"/":r.length>0?r:"."},normalize:function(e){if(n(e),0===e.length)return".";var r=47===e.charCodeAt(0),i=47===e.charCodeAt(e.length-1);return 0!==(e=t(e,!r)).length||r||(e="."),e.length>0&&i&&(e+="/"),r?"/"+e:e},isAbsolute:function(e){return n(e),e.length>0&&47===e.charCodeAt(0)},join:function(){if(0===arguments.length)return".";for(var e,t=0;t<arguments.length;++t){var i=arguments[t];n(i),i.length>0&&(void 0===e?e=i:e+="/"+i)}return void 0===e?".":r.normalize(e)},relative:function(e,t){if(n(e),n(t),e===t)return"";if((e=r.resolve(e))===(t=r.resolve(t)))return"";for(var i=1;i<e.length&&47===e.charCodeAt(i);++i);for(var o=e.length,l=o-i,c=1;c<t.length&&47===t.charCodeAt(c);++c);for(var s=t.length-c,a=l<s?l:s,d=-1,u=0;u<=a;++u){if(u===a){if(s>a){if(47===t.charCodeAt(c+u))return t.slice(c+u+1);if(0===u)return t.slice(c+u)}else l>a&&(47===e.charCodeAt(i+u)?d=u:0===u&&(d=0));break}var f=e.charCodeAt(i+u);if(f!==t.charCodeAt(c+u))break;47===f&&(d=u)}var h="";for(u=i+d+1;u<=o;++u)u!==o&&47!==e.charCodeAt(u)||(0===h.length?h+="..":h+="/..");return h.length>0?h+t.slice(c+d):(c+=d,47===t.charCodeAt(c)&&++c,t.slice(c))},_makeLong:function(e){return e},dirname:function(e){if(n(e),0===e.length)return".";for(var t=e.charCodeAt(0),r=47===t,i=-1,o=!0,l=e.length-1;l>=1;--l)if(47===(t=e.charCodeAt(l))){if(!o){i=l;break}}else o=!1;return-1===i?r?"/":".":r&&1===i?"//":e.slice(0,i)},basename:function(e,t){if(void 0!==t&&"string"!==typeof t)throw new TypeError('"ext" argument must be a string');n(e);var r,i=0,o=-1,l=!0;if(void 0!==t&&t.length>0&&t.length<=e.length){if(t.length===e.length&&t===e)return"";var c=t.length-1,s=-1;for(r=e.length-1;r>=0;--r){var a=e.charCodeAt(r);if(47===a){if(!l){i=r+1;break}}else-1===s&&(l=!1,s=r+1),c>=0&&(a===t.charCodeAt(c)?-1===--c&&(o=r):(c=-1,o=s))}return i===o?o=s:-1===o&&(o=e.length),e.slice(i,o)}for(r=e.length-1;r>=0;--r)if(47===e.charCodeAt(r)){if(!l){i=r+1;break}}else-1===o&&(l=!1,o=r+1);return-1===o?"":e.slice(i,o)},extname:function(e){n(e);for(var t=-1,r=0,i=-1,o=!0,l=0,c=e.length-1;c>=0;--c){var s=e.charCodeAt(c);if(47!==s)-1===i&&(o=!1,i=c+1),46===s?-1===t?t=c:1!==l&&(l=1):-1!==t&&(l=-1);else if(!o){r=c+1;break}}return-1===t||-1===i||0===l||1===l&&t===i-1&&t===r+1?"":e.slice(t,i)},format:function(e){if(null===e||"object"!==typeof e)throw new TypeError('The "pathObject" argument must be of type Object. Received type '+typeof e);return function(e,n){var t=n.dir||n.root,r=n.base||(n.name||"")+(n.ext||"");return t?t===n.root?t+r:t+e+r:r}("/",e)},parse:function(e){n(e);var t={root:"",dir:"",base:"",ext:"",name:""};if(0===e.length)return t;var r,i=e.charCodeAt(0),o=47===i;o?(t.root="/",r=1):r=0;for(var l=-1,c=0,s=-1,a=!0,d=e.length-1,u=0;d>=r;--d)if(47!==(i=e.charCodeAt(d)))-1===s&&(a=!1,s=d+1),46===i?-1===l?l=d:1!==u&&(u=1):-1!==l&&(u=-1);else if(!a){c=d+1;break}return-1===l||-1===s||0===u||1===u&&l===s-1&&l===c+1?-1!==s&&(t.base=t.name=0===c&&o?e.slice(1,s):e.slice(c,s)):(0===c&&o?(t.name=e.slice(1,l),t.base=e.slice(1,s)):(t.name=e.slice(c,l),t.base=e.slice(c,s)),t.ext=e.slice(l,s)),c>0?t.dir=e.slice(0,c-1):o&&(t.dir="/"),t},sep:"/",delimiter:":",win32:null,posix:null};r.posix=r,e.exports=r}},t={};function r(e){var i=t[e];if(void 0!==i)return i.exports;var o=t[e]={exports:{}},l=!0;try{n[e](o,o.exports,r),l=!1}finally{l&&delete t[e]}return o.exports}r.ab="//";var i=r(114);e.exports=i}()},58146:function(e,n,t){"use strict";var r=t(75582),i=t(82394),o=t(21764),l=t(82684),c=t(69864),s=t(34376),a=t(71180),d=t(15338),u=t(97618),f=t(55485),h=t(85854),g=t(65956),p=t(36288),x=t(44085),m=t(38276),v=t(30160),E=t(35576),Z=t(17488),j=t(69650),A=t(35686),b=t(8193),I=t(72473),L=t(70515),_=t(48277),D=t(81728),y=t(3917),P=t(76417),R=t(42122),T=t(72619),w=t(28598);function S(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function C(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?S(Object(t),!0).forEach((function(n){(0,i.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):S(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var k=2*L.iI;n.Z=function(e){var n=e.onCancel,t=e.permission,i=(0,s.useRouter)(),S=(0,l.useState)({}),B=S[0],O=S[1],W=(0,l.useState)(null),N=W[0],U=W[1],H=(0,l.useCallback)((function(e){O((function(n){return C(C({},n),e)})),U((function(n){return C(C({},n),e)}))}),[O,U]);(0,l.useEffect)((function(){t&&U(t)}),[U,t]);var M=A.ZP.permissions.list({_format:"with_only_entity_options",only_entity_options:!0},{},{pauseFetch:!!t}).data,Q=(0,l.useMemo)((function(){var e;return null===M||void 0===M||null===(e=M.permissions)||void 0===e?void 0:e[0]}),[M]),V=(0,l.useMemo)((function(){var e;return(null===(e=t||Q)||void 0===e?void 0:e.entity_names)||[]}),[t,Q]),Y=(0,l.useMemo)((function(){var e;return(null===(e=t||Q)||void 0===e?void 0:e.entity_types)||[]}),[t,Q]),z=(0,c.Db)(t?A.ZP.permissions.useUpdate(null===t||void 0===t?void 0:t.id):A.ZP.permissions.useCreate(),{onSuccess:function(e){return(0,T.wD)(e,{callback:function(e){var n=e.permission;O({}),U(n),t||i.push("/settings/workspace/permissions/".concat(null===n||void 0===n?void 0:n.id)),o.Am.success(t?"Permission successfully updated.":"New permission created successfully.",{position:o.Am.POSITION.BOTTOM_RIGHT,toastId:"permission-mutate-success-".concat(n.id)})},onErrorCallback:function(e){var n=e.error,t=n.errors,r=n.exception,i=n.message,l=n.type;o.Am.error((null===t||void 0===t?void 0:t.error)||r||i,{position:o.Am.POSITION.BOTTOM_RIGHT,toastId:l})}})}}),q=(0,r.Z)(z,2),F=q[0],G=q[1].isLoading,J=(0,c.Db)(A.ZP.permissions.useDelete(null===t||void 0===t?void 0:t.id),{onSuccess:function(e){return(0,T.wD)(e,{callback:function(){i.push("/settings/workspace/permissions"),o.Am.success("Permission successfully delete.",{position:o.Am.POSITION.BOTTOM_RIGHT,toastId:"permission-delete-success-".concat(null===t||void 0===t?void 0:t.id)})},onErrorCallback:function(e){var n=e.error,t=n.errors,r=n.exception,i=n.message,l=n.type;o.Am.error((null===t||void 0===t?void 0:t.error)||r||i,{position:o.Am.POSITION.BOTTOM_RIGHT,toastId:l})}})}}),$=(0,r.Z)(J,2),K=$[0],X=$[1].isLoading,ee=(0,l.useMemo)((function(){return(null===N||void 0===N?void 0:N.access)||0}),[N]),ne=(0,l.useCallback)((function(e){return e.map((function(e,n){var t=p.K4[e],r=Boolean(ee&Number(e)),i=(0,_.fD)(ee),o=(0,_.fD)(e);return(0,w.jsx)(m.Z,{mt:n>=1?1:0,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(j.Z,{checked:r,compact:!0,onCheck:function(e){return H({access:(0,_.$P)(e(r)?(0,_.vN)(i,o):(0,_.VJ)(i,o))})}}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(v.ZP,{default:!r,children:t})]})},t)}))}),[ee]),te=(0,l.useMemo)((function(){return(null===t||void 0===t?void 0:t.roles)||[]}),[t]),re=(0,l.useMemo)((function(){return(null===t||void 0===t?void 0:t.users)||[]}),[t]),ie=(0,l.useMemo)((function(){return(null===te||void 0===te?void 0:te.length)>=1}),[te]),oe=(0,l.useMemo)((function(){return(null===re||void 0===re?void 0:re.length)>=1}),[re]);return(0,w.jsxs)(b.N,{children:[(0,w.jsxs)(g.Z,{noPadding:!0,children:[(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsx)(h.Z,{level:4,children:t?"Permission ".concat(null===t||void 0===t?void 0:t.id):"New permission"})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsxs)(v.ZP,{danger:"entity_name"in B&&!(null!==N&&void 0!==N&&N.entity_name),default:!0,large:!0,children:["Entity ","entity_name"in B&&!(null!==N&&void 0!==N&&N.entity_name)&&(0,w.jsx)(v.ZP,{danger:!0,inline:!0,large:!0,children:"is required"})]}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(u.Z,{flex:1,justifyContent:"flex-end",children:(0,w.jsx)(x.Z,{afterIconSize:k,alignRight:!0,autoComplete:"off",large:!0,noBackground:!0,noBorder:!0,onChange:function(e){return H({entity_name:e.target.value})},paddingHorizontal:0,paddingVertical:0,placeholder:"Select an entity",value:(null===N||void 0===N?void 0:N.entity_name)||"",children:V.map((function(e){return(0,w.jsx)("option",{value:e,children:(0,D.j3)(e)},e)}))})})]})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Entity subtype"}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(u.Z,{flex:1,justifyContent:"flex-end",children:(0,w.jsxs)(x.Z,{afterIconSize:k,alignRight:!0,autoComplete:"off",large:!0,monospace:!0,noBackground:!0,noBorder:!0,onChange:function(e){return H({entity_type:e.target.value})},paddingHorizontal:0,paddingVertical:0,placeholder:"Select an entity subtype",value:(null===N||void 0===N?void 0:N.entity_type)||"",children:[(0,w.jsx)("option",{value:""}),Y.map((function(e){return(0,w.jsx)("option",{value:e,children:e},e)}))]})})]})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Enity UUID"}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsx)(Z.Z,{afterIcon:(0,w.jsx)(I.I8,{}),afterIconClick:function(e,n){var t;null===n||void 0===n||null===(t=n.current)||void 0===t||t.focus()},afterIconSize:k,alignRight:!0,autoComplete:"off",large:!0,monospace:!0,noBackground:!0,noBorder:!0,fullWidth:!0,onChange:function(e){return H({entity_id:e.target.value})},paddingHorizontal:0,paddingVertical:0,placeholder:"e.g. pipeline_uuid",value:(null===N||void 0===N?void 0:N.entity_id)||""})})]})})]}),(0,w.jsx)(m.Z,{mb:L.HN}),(0,w.jsxs)(g.Z,{noPadding:!0,children:[(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsx)(h.Z,{level:4,children:"Access"})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsxs)(m.Z,{p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Groups"})}),ne(p.G9)]}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Operations"})}),ne(p.Pt)]})}),(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Disable operations"})}),ne(p.oO)]})})]}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsxs)(f.ZP,{alignItems:"flex-start",children:[(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Read attributes"})}),ne(p.Fy)]})}),(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{fullWidth:!0,p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Readable attributes (comma separated)"})}),(0,w.jsx)(E.Z,{fullWidth:!0,monospace:!0,onChange:function(e){return H({read_attributes:e.target.value})},placeholder:"e.g. email",value:(null===N||void 0===N?void 0:N.read_attributes)||""})]})})]}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsxs)(f.ZP,{alignItems:"flex-start",children:[(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Write attributes"})}),ne(p.H1)]})}),(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{fullWidth:!0,p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Writable attributes (comma separated)"})}),(0,w.jsx)(E.Z,{fullWidth:!0,monospace:!0,onChange:function(e){return H({write_attributes:e.target.value})},placeholder:"e.g. password",value:(null===N||void 0===N?void 0:N.write_attributes)||""})]})})]}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsxs)(f.ZP,{alignItems:"flex-start",children:[(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Query parameters"})}),ne(p.hl)]})}),(0,w.jsx)(u.Z,{flex:1,children:(0,w.jsxs)(m.Z,{fullWidth:!0,p:L.cd,children:[(0,w.jsx)(m.Z,{mb:L.cd,children:(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Parameters that can be queried (comma separated)"})}),(0,w.jsx)(E.Z,{fullWidth:!0,monospace:!0,onChange:function(e){return H({query_attributes:e.target.value})},placeholder:"e.g. include_outputs",value:(null===N||void 0===N?void 0:N.query_attributes)||""})]})})]})]}),(0,w.jsx)(m.Z,{mb:L.HN}),t&&(0,w.jsxs)(w.Fragment,{children:[(0,w.jsxs)(g.Z,{noPadding:!0,children:[(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsx)(f.ZP,{alignItems:"center",justifyContent:"space-between",children:(0,w.jsx)(h.Z,{level:4,children:"Roles"})})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:!ie&&(0,w.jsx)(v.ZP,{default:!0,children:"This permission is currently not attached to any role."})})]}),(0,w.jsx)(m.Z,{mb:L.HN}),(0,w.jsxs)(g.Z,{noPadding:!0,children:[(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsx)(f.ZP,{alignItems:"center",justifyContent:"space-between",children:(0,w.jsx)(h.Z,{level:4,children:"Users"})})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:!oe&&(0,w.jsx)(v.ZP,{default:!0,children:"There are currently no users with this permission."})})]}),(0,w.jsx)(m.Z,{mb:L.HN}),(0,w.jsxs)(g.Z,{noPadding:!0,children:[(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsx)(h.Z,{level:4,children:"Metadata"})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Last updated"}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsxs)(u.Z,{alignItems:"center",flex:1,justifyContent:"flex-end",children:[(0,w.jsx)(v.ZP,{large:!0,monospace:!0,muted:!0,children:(null===N||void 0===N?void 0:N.updated_at)&&(0,y.d$)(null===N||void 0===N?void 0:N.updated_at,{includeSeconds:!0})}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(I.Pf,{muted:!0,size:k}),(0,w.jsx)(m.Z,{mr:1})]})]})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Created at"}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsxs)(u.Z,{alignItems:"center",flex:1,justifyContent:"flex-end",children:[(0,w.jsx)(v.ZP,{large:!0,monospace:!0,muted:!0,children:(null===N||void 0===N?void 0:N.created_at)&&(0,y.d$)(null===N||void 0===N?void 0:N.created_at,{includeSeconds:!0})}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(I.Pf,{muted:!0,size:k}),(0,w.jsx)(m.Z,{mr:1})]})]})}),(0,w.jsx)(d.Z,{light:!0}),(0,w.jsx)(m.Z,{p:L.cd,children:(0,w.jsxs)(f.ZP,{alignItems:"center",children:[(0,w.jsx)(v.ZP,{default:!0,large:!0,children:"Created by"}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsxs)(u.Z,{alignItems:"center",flex:1,justifyContent:"flex-end",children:[(0,w.jsx)(v.ZP,{large:!0,monospace:!0,muted:!0,children:(0,P.s)(null===t||void 0===t?void 0:t.user)}),(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(I.Pf,{muted:!0,size:k}),(0,w.jsx)(m.Z,{mr:1})]})]})})]}),(0,w.jsx)(m.Z,{mb:L.HN})]}),(0,w.jsxs)(f.ZP,{children:[(0,w.jsx)(a.Z,{beforeIcon:(0,w.jsx)(I.vc,{}),disabled:!B||(0,R.Qr)(B),loading:G,onClick:function(){return F({permission:(0,R.GL)(N,["access","entity_id","entity_name","entity_type","query_attributes","read_attributes","write_attributes"],{include_blanks:!0})})},primary:!0,children:t?"Save changes":"Create new permission"}),n&&(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(a.Z,{onClick:function(){return null===n||void 0===n?void 0:n()},secondary:!0,children:"Cancel and go back"})]}),t&&(0,w.jsxs)(w.Fragment,{children:[(0,w.jsx)(m.Z,{mr:L.cd}),(0,w.jsx)(a.Z,{beforeIcon:(0,w.jsx)(I.rF,{}),danger:!0,loading:X,onClick:function(){return K()},children:"Delete permission"})]})]})]})}},8193:function(e,n,t){"use strict";t.d(n,{N:function(){return c}});var r=t(38626),i=t(44897),o=t(42631),l=t(70515),c=r.default.div.withConfig({displayName:"indexstyle__ContainerStyle",componentId:"sc-1ck7mzt-0"})(["border-radius:","px;padding:","px;",""],o.n_,l.cd*l.iI,(function(e){return"\n    background-color: ".concat((e.theme.background||i.Z.background).codeArea,";\n  ")}))},36288:function(e,n,t){"use strict";t.d(n,{Fy:function(){return u},G9:function(){return c},H1:function(){return f},K4:function(){return l},Pt:function(){return s},hl:function(){return d},oO:function(){return a}});var r,i,o=t(82394);!function(e){e[e.OWNER=1]="OWNER",e[e.ADMIN=2]="ADMIN",e[e.EDITOR=4]="EDITOR",e[e.VIEWER=8]="VIEWER",e[e.LIST=16]="LIST",e[e.DETAIL=32]="DETAIL",e[e.CREATE=64]="CREATE",e[e.UPDATE=128]="UPDATE",e[e.DELETE=512]="DELETE",e[e.OPERATION_ALL=1024]="OPERATION_ALL",e[e.QUERY=2048]="QUERY",e[e.QUERY_ALL=4096]="QUERY_ALL",e[e.READ=8192]="READ",e[e.READ_ALL=16384]="READ_ALL",e[e.WRITE=32768]="WRITE",e[e.WRITE_ALL=65536]="WRITE_ALL",e[e.ALL=131072]="ALL",e[e.DISABLE_LIST=262144]="DISABLE_LIST",e[e.DISABLE_DETAIL=524288]="DISABLE_DETAIL",e[e.DISABLE_CREATE=1048576]="DISABLE_CREATE",e[e.DISABLE_UPDATE=2097152]="DISABLE_UPDATE",e[e.DISABLE_DELETE=4194304]="DISABLE_DELETE",e[e.DISABLE_OPERATION_ALL=8388608]="DISABLE_OPERATION_ALL",e[e.DISABLE_QUERY=16777216]="DISABLE_QUERY",e[e.DISABLE_QUERY_ALL=33554432]="DISABLE_QUERY_ALL",e[e.DISABLE_READ=67108864]="DISABLE_READ",e[e.DISABLE_READ_ALL=134217728]="DISABLE_READ_ALL",e[e.DISABLE_WRITE=268435456]="DISABLE_WRITE",e[e.DISABLE_WRITE_ALL=536870912]="DISABLE_WRITE_ALL"}(i||(i={}));var l=(r={},(0,o.Z)(r,i.OWNER,"Owner"),(0,o.Z)(r,i.ADMIN,"Admin"),(0,o.Z)(r,i.EDITOR,"Editor"),(0,o.Z)(r,i.VIEWER,"Viewer"),(0,o.Z)(r,i.LIST,"List"),(0,o.Z)(r,i.DETAIL,"Detail"),(0,o.Z)(r,i.CREATE,"Create"),(0,o.Z)(r,i.UPDATE,"Update"),(0,o.Z)(r,i.DELETE,"Delete"),(0,o.Z)(r,i.OPERATION_ALL,"All operations"),(0,o.Z)(r,i.QUERY,"Query"),(0,o.Z)(r,i.QUERY_ALL,"Query all attributes"),(0,o.Z)(r,i.READ,"Read"),(0,o.Z)(r,i.READ_ALL,"Read all attributes"),(0,o.Z)(r,i.WRITE,"Write"),(0,o.Z)(r,i.WRITE_ALL,"Write all attributes"),(0,o.Z)(r,i.ALL,"All"),(0,o.Z)(r,i.DISABLE_LIST,"Disable list"),(0,o.Z)(r,i.DISABLE_DETAIL,"Disable detail"),(0,o.Z)(r,i.DISABLE_CREATE,"Disable create"),(0,o.Z)(r,i.DISABLE_UPDATE,"Disable update"),(0,o.Z)(r,i.DISABLE_DELETE,"Disable delete"),(0,o.Z)(r,i.DISABLE_OPERATION_ALL,"Disable all operations"),(0,o.Z)(r,i.DISABLE_QUERY,"Disable query"),(0,o.Z)(r,i.DISABLE_QUERY_ALL,"Disable all query parameters"),(0,o.Z)(r,i.DISABLE_READ,"Disable read"),(0,o.Z)(r,i.DISABLE_READ_ALL,"Disable all read attributes"),(0,o.Z)(r,i.DISABLE_WRITE,"Disable write"),(0,o.Z)(r,i.DISABLE_WRITE_ALL,"Disable all write attributes"),r),c=[i.OWNER,i.ADMIN,i.EDITOR,i.VIEWER,i.ALL],s=[i.LIST,i.DETAIL,i.CREATE,i.UPDATE,i.DELETE,i.OPERATION_ALL],a=[i.DISABLE_LIST,i.DISABLE_DETAIL,i.DISABLE_CREATE,i.DISABLE_UPDATE,i.DISABLE_DELETE,i.DISABLE_OPERATION_ALL],d=[i.QUERY,i.QUERY_ALL,i.DISABLE_QUERY,i.DISABLE_QUERY_ALL],u=[i.READ,i.READ_ALL,i.DISABLE_READ,i.DISABLE_READ_ALL],f=[i.WRITE,i.WRITE_ALL,i.DISABLE_WRITE,i.DISABLE_WRITE_ALL]},65956:function(e,n,t){"use strict";var r=t(38626),i=t(55485),o=t(38276),l=t(30160),c=t(44897),s=t(42631),a=t(47041),d=t(70515),u=t(28598),f=(0,r.css)(["padding:","px;padding-bottom:","px;padding-top:","px;"],2*d.iI,1.5*d.iI,1.5*d.iI),h=r.default.div.withConfig({displayName:"Panel__PanelStyle",componentId:"sc-1ct8cgl-0"})(["border-radius:","px;overflow:hidden;"," "," "," "," "," "," "," "," "," "," "," ",""],s.n_,(function(e){return e.fullWidth&&"\n    width: 100%;\n  "}),(function(e){return!e.borderless&&"\n    border: 1px solid ".concat((e.theme.interactive||c.Z.interactive).defaultBorder,";\n  ")}),(function(e){return e.success&&"\n    background-color: ".concat((e.theme.background||c.Z.background).successLight,";\n  ")}),(function(e){return e.success&&!e.borderless&&"\n    border: 1px solid ".concat((e.theme.background||c.Z.background).success,";\n  ")}),(function(e){return!e.dark&&!e.success&&"\n    background-color: ".concat((e.theme.background||c.Z.background).panel,";\n  ")}),(function(e){return e.dark&&"\n    background-color: ".concat((e.theme.background||c.Z.background).content,";\n  ")}),(function(e){return!e.fullHeight&&"\n    height: fit-content;\n  "}),(function(e){return e.maxHeight&&"\n    max-height: ".concat(e.maxHeight,";\n  ")}),(function(e){return e.maxWidth&&"\n    max-width: ".concat(e.maxWidth,"px;\n  ")}),(function(e){return e.minWidth&&"\n    min-width: ".concat(e.minWidth,"px;\n\n    @media (max-width: ").concat(e.minWidth,"px) {\n      min-width: 0;\n    }\n  ")}),(function(e){return e.borderless&&"\n    border: none;\n  "}),(function(e){return e.overflowVisible&&"\n    overflow: visible;\n  "})),g=r.default.div.withConfig({displayName:"Panel__HeaderStyle",componentId:"sc-1ct8cgl-1"})(["border-top-left-radius:","px;border-top-right-radius:","px;"," "," "," ",""],s.n_,s.n_,(function(e){return"\n    background-color: ".concat((e.theme.background||c.Z.background).chartBlock,";\n    border-bottom: 1px solid ").concat((e.theme.interactive||c.Z.interactive).defaultBorder,";\n  ")}),(function(e){return e.height&&"\n    height: ".concat(e.height,"px;\n  ")}),f,(function(e){return e.headerPaddingVertical&&"\n    padding-bottom: ".concat(e.headerPaddingVertical,"px;\n    padding-top: ").concat(e.headerPaddingVertical,"px;\n  ")})),p=r.default.div.withConfig({displayName:"Panel__ContentStyle",componentId:"sc-1ct8cgl-2"})(["overflow-y:auto;padding:","px;height:100%;"," "," "," "," ",""],1.75*d.iI,a.w5,(function(e){return e.height&&"\n    height: ".concat(e.height,"px;\n  ")}),(function(e){return e.maxHeight&&"\n    max-height: calc(".concat(e.maxHeight," - ").concat(15*d.iI,"px);\n  ")}),(function(e){return e.noPadding&&"\n    padding: 0;\n  "}),(function(e){return e.overflowVisible&&"\n    overflow: visible;\n  "})),x=r.default.div.withConfig({displayName:"Panel__FooterStyle",componentId:"sc-1ct8cgl-3"})(["border-style:",";border-top-width:","px;padding:","px;"],s.M8,s.YF,1.75*d.iI);n.Z=function(e){var n=e.borderless,t=e.children,r=e.containerRef,c=e.contentContainerRef,s=e.dark,a=e.footer,d=e.fullHeight,f=void 0===d||d,m=e.fullWidth,v=void 0===m||m,E=e.header,Z=e.headerHeight,j=e.headerIcon,A=e.headerPaddingVertical,b=e.headerTitle,I=e.maxHeight,L=e.maxWidth,_=e.minWidth,D=e.noPadding,y=e.overflowVisible,P=e.subtitle,R=e.success;return(0,u.jsxs)(h,{borderless:n,dark:s,fullHeight:f,fullWidth:v,maxHeight:I,maxWidth:L,minWidth:_,overflowVisible:y,ref:r,success:R,children:[(E||b)&&(0,u.jsxs)(g,{headerPaddingVertical:A,height:Z,children:[E&&E,b&&(0,u.jsx)(i.ZP,{alignItems:"center",justifyContent:"space-between",children:(0,u.jsxs)(i.ZP,{alignItems:"center",children:[j&&j,(0,u.jsx)(o.Z,{ml:j?1:0,children:(0,u.jsx)(l.ZP,{bold:!0,default:!0,children:b})})]})})]}),(0,u.jsxs)(p,{maxHeight:I,noPadding:D,overflowVisible:y,ref:c,children:[P&&(0,u.jsx)(o.Z,{mb:2,children:(0,u.jsx)(l.ZP,{default:!0,children:P})}),t]}),a&&(0,u.jsx)(x,{children:a})]})}},76417:function(e,n,t){"use strict";function r(e){return null!==e&&void 0!==e&&e.first_name?[null===e||void 0===e?void 0:e.first_name,null===e||void 0===e?void 0:e.last_name].filter((function(e){return e})).join(" "):null===e||void 0===e?void 0:e.username}t.d(n,{s:function(){return r}})},48277:function(e,n,t){"use strict";t.d(n,{$P:function(){return a},JI:function(){return o},VJ:function(){return s},fD:function(){return l},uf:function(){return i},vN:function(){return c}});var r=t(75582),i=function(e){var n=String(e).split("."),t=(0,r.Z)(n,2),i=t[0],o=t[1];return"".concat(i.replace(/\B(?=(\d{3})+(?!\d))/g,",")).concat(o?".".concat(o):"")};function o(e){var n=Math.floor(Date.now()/1e3);return e>0?n-e:n}function l(e){return(e>>>0).toString(2)}function c(e,n){return String(BigInt(e)+BigInt(n))}function s(e,n){return String(BigInt(e)-BigInt(n))}function a(e){return parseInt(e,2)}},81728:function(e,n,t){"use strict";t.d(n,{RA:function(){return h},j3:function(){return A},kC:function(){return g},vg:function(){return j},kE:function(){return P},Mp:function(){return p},Pb:function(){return d},HW:function(){return I},HD:function(){return u},wX:function(){return x},x6:function(){return m},_6:function(){return v},zf:function(){return b},Y6:function(){return D},Lo:function(){return y},wE:function(){return R},Tz:function(){return L},J3:function(){return E},We:function(){return f},QV:function(){return _},C5:function(){return Z}});var r=t(75582),i=t(17717),o=["aged","ancient","autumn","billowing","bitter","black","blue","bold","broken","cold","cool","crimson","damp","dark","dawn","delicate","divine","dry","empty","falling","floral","fragrant","frosty","green","hidden","holy","icy","late","lingering","little","lively","long","misty","morning","muddy","nameless","old","patient","polished","proud","purple","quiet","red","restless","rough","shy","silent","small","snowy","solitary","sparkling","spring","still","summer","throbbing","twilight","wandering","weathered","white","wild","winter","wispy","withered","young"],l=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"],c=(t(92083),["bird","breeze","brook","bush","butterfly","cherry","cloud","darkness","dawn","dew","dream","dust","feather","field","fire","firefly","flower","fog","forest","frog","frost","glade","glitter","grass","haze","hill","lake","leaf","meadow","moon","morning","mountain","night","paper","pine","pond","rain","resonance","river","sea","shadow","shape","silence","sky","smoke","snow","snowflake","sound","star","sun","sun","sunset","surf","thunder","tree","violet","voice","water","water","waterfall","wave","wildflower","wind","wood"]),s=["0","1","2","3","4","5","6","7","8","9"],a=t(86735);function d(e){if(!e)return!1;try{JSON.parse(e)}catch(n){return!1}return!0}function u(e){return"string"===typeof e}function f(e){var n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"_";return e.split(" ").join(n)}function h(e){return e.split(" ").join("_")}function g(e){return e?e.charAt(0).toUpperCase()+e.slice(1):""}function p(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:1;return String((new Date).getTime()*e)}function x(e){return e.charAt(0).toLowerCase()+e.slice(1)}function m(e){if(null===e||"undefined"===typeof e)return"";var n=e.toString().split("."),t=(0,r.Z)(n,2),i=t[0],o=t[1],l=i.toString().replace(/\B(?=(\d{3})+(?!\d))/g,",");return o?"".concat(l,".").concat(o):l}function v(e,n){var t,r=arguments.length>2&&void 0!==arguments[2]&&arguments[2],i=arguments.length>3&&void 0!==arguments[3]&&arguments[3],o=n,l=void 0!==o&&null!==o;if(l||(o=2),1===o)t=e;else{var c=e.length,s=e[c-1];t="y"===s&&"day"!==e?"".concat(e.slice(0,c-1),"ies"):"".concat(e,"s"===s?"es":"s")}if(l&&!i){var a=r?m(o):o;return"".concat(a," ").concat(t)}return t}function E(e){return null===e||void 0===e?void 0:e.replace(/_/g," ")}function Z(e){var n=e.length;return"ies"===e.slice(n-3,n)?"".concat(e.slice(0,n-3),"y"):"es"===e.slice(n-2,n)&&"ces"!==e.slice(n-3,n)?e.slice(0,n-2):e.slice(0,n-1)}function j(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"";return g(E(e.toLowerCase()))}function A(e){return e.replace(/([A-Z])/g," $1")}function b(e){var n,t=[["second",60],["minute",60],["hour",24],["day",7],["week",4],["month",12],["year",null]];return t.forEach((function(i,o){if(!n){var l=(0,r.Z)(i,2),c=l[0],s=l[1],a=t.slice(0,o).reduce((function(e,n){return e*Number(n[1])}),1);e<Number(s)*a&&(n=v(c,Math.round(e/a)))}})),n}function I(e){return"undefined"!==typeof e&&null!==e&&!isNaN(e)}function L(e){var n,t=e.match(/\d+(\.?\d*)%/)||[];return Number(null===(n=t[0])||void 0===n?void 0:n.slice(0,-1))}function _(e){var n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:2,t=Math.pow(10,n);return Math.round((e||0)*t)/t}function D(){return"".concat((0,a.mp)(o)," ").concat((0,a.mp)(c))}function y(){return"".concat((0,a.mp)(l)).concat((0,a.mp)(s))}function P(e){return null===e||void 0===e?void 0:e.toLowerCase().replace(/\W+/g,"_")}function R(e){var n,t=e.split(i.sep),r=t[t.length-1].split(".");return n=1===r.length?r[0]:r.slice(0,-1).join("."),t.slice(0,t.length-1).concat(n).join(i.sep)}}}]);