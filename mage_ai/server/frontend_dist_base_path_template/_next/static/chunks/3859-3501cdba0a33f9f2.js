"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[3859],{93859:function(e,n,t){t.d(n,{r:function(){return B},Z:function(){return P}});var r=t(82394),o=t(21831),i=t(82684),a=t(48670),c=t(12691),l=t.n(c),u=t(38626),d=t(78141),s=t(44628),h=t(6508),m=t(55485),f=t(30160),p=t(44897),g=t(70987),v=function(){var e=document.createElement("div");e.setAttribute("style","width: 100px; height: 100px; overflow: scroll; position:absolute; top:-9999px;"),document.body.appendChild(e);var n=e.offsetWidth-e.clientWidth;return document.body.removeChild(e),n},b=t(95363),y=t(61896),x=t(47041),w=t(48888),H=t(70515),S=t(40489),j=t(86735),O=t(28598);function k(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function E(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?k(Object(t),!0).forEach((function(n){(0,r.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):k(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var Z=2*H.iI+y.dN,A=20*H.iI,B=8.7,C=u.default.div.withConfig({displayName:"DataTable__Styles",componentId:"sc-1arr863-0"})([""," "," "," .body > div{","}.table{border-spacing:0;display:inline-block;"," "," "," "," .tr{.td.td-index-column{","}}.th{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;","}.th,.td{"," font-family:",";margin:0;","}.td{padding:","px;}&.sticky{overflow:auto;}.header{overflow:hidden;}}"],(function(e){return e.disableScrolling&&"\n    overflow: hidden;\n  "}),(function(e){return e.height&&"\n    height: ".concat(e.height,"px;\n  ")}),(function(e){return e.maxHeight&&"\n    max-height: ".concat(e.maxHeight,"px;\n  ")}),x.w5,(function(e){return!e.noBorderBottom&&"\n      border-bottom: 1px solid ".concat((e.theme.borders||g.Z.borders).medium,";\n    ")}),(function(e){return!e.noBorderLeft&&"\n      border-left: 1px solid ".concat((e.theme.borders||g.Z.borders).medium,";\n    ")}),(function(e){return!e.noBorderRight&&"\n      border-right: 1px solid ".concat((e.theme.borders||g.Z.borders).medium,";\n    ")}),(function(e){return!e.noBorderTop&&"\n      border-top: 1px solid ".concat((e.theme.borders||g.Z.borders).medium,";\n    ")}),(function(e){return"\n          color: ".concat((e.theme.content||g.Z.content).default,";\n        ")}),(function(e){return"\n        height: ".concat(e.columnHeaderHeight||Z,"px;\n      ")}),y.iD,b.ry,(function(e){return"\n        background-color: ".concat((e.theme.background||g.Z.background).table,";\n        border-bottom: 1px solid ").concat((e.theme.borders||g.Z.borders).medium,";\n        border-right: 1px solid ").concat((e.theme.borders||g.Z.borders).medium,";\n      ")}),1*H.iI);function M(e){var n=e.original,t=8.5*Math.max.apply(Math,(0,o.Z)(n.map((function(e){return(null===e||void 0===e?void 0:e.length)||0})))),r=Math.ceil(t/(A-2*H.iI));return Math.max(r,1)*y.dN+2*H.iI}function I(e){var n=e.columnHeaderHeight,t=e.columns,r=e.data,c=e.disableScrolling,x=e.height,k=e.index,C=e.invalidValues,I=e.maxHeight,P=e.numberOfIndexes,T=e.previewIndexes,N=e.renderColumnHeader,_=e.width,R=(0,i.useContext)(u.ThemeContext),D=(0,i.useRef)(null),L=(0,i.useRef)(null);(0,i.useEffect)((function(){var e=function(e){var n;null===D||void 0===D||null===(n=D.current)||void 0===n||n.scroll(e.target.scrollLeft,0)};return L&&L.current.addEventListener("scroll",e),function(){var n;null===L||void 0===L||null===(n=L.current)||void 0===n||n.removeEventListener("scroll",e)}}),[D,L]);var V=(0,i.useMemo)((function(){return k&&r&&k.length===r.length}),[r,k]),W=(0,i.useMemo)((function(){var e=[];return(0,j.w6)(P).forEach((function(n,t){var i=String(null===r||void 0===r?void 0:r.length).length*B;if(V){var a=k.map((function(e){return P>=2?String(e[t]).length:String(e).length}));i=Math.max.apply(Math,(0,o.Z)(a))*B}e.push(i+2*H.iI)})),e}),[r,k,P,V]),z=t.map((function(e){return null===e||void 0===e?void 0:e.Header})).slice(1),F=(0,i.useMemo)((function(){return v()}),[]),G=(0,i.useMemo)((function(){var e=_-(Math.max.apply(Math,(0,o.Z)(W))+1.5*H.iI+F),n=t.length-1,r=A;return r*n<e&&(r=e/n),{width:r}}),[t,W,F,_]),U=(0,s.useTable)({columns:t,data:r,defaultColumn:G},s.useBlockLayout,h.useSticky),q=U.getTableBodyProps,J=U.getTableProps,X=U.headerGroups,K=U.prepareRow,Q=U.rows,Y=(0,i.useCallback)((function(e){var n=e.index,t=e.style,r=new Set((null===T||void 0===T?void 0:T.removedRows)||[]),o=Q[n];K(o);var c=o.original,u=r.has(n);return(0,O.jsx)("div",E(E({},o.getRowProps({style:E(E({},t),{},{width:"auto"})})),{},{className:"tr",children:o.cells.map((function(e,t){var r,o=t<=P-1,d=e.getCellProps(),s=e.column.id,h=null===C||void 0===C||null===(r=C[s])||void 0===r?void 0:r.includes(n),p=E({},d.style);o&&(p.fontFamily=b.Vp,p.left=0,p.position="sticky",p.textAlign=k?"right":"center",p.width=W[t]);var v,y=c[t-P],x=z.indexOf(s);if(h&&(p.color=g.Z.interactive.dangerBorder),u&&(p.backgroundColor=g.Z.background.danger),Array.isArray(y)||"object"===typeof y)try{y=JSON.stringify(y)}catch(H){y="Error: cannot display value"}return o&&(V?(v=k[n],Array.isArray(v)&&(v=v[t])):v=e.render("Cell")),(0,i.createElement)("div",E(E({},d),{},{className:"td ".concat(o?"td-index-column":""),key:"".concat(t,"-").concat(y),style:p}),v,!o&&(0,O.jsxs)(m.ZP,{justifyContent:"space-between",children:[(0,O.jsxs)(f.ZP,{danger:h,default:!0,wordBreak:!0,children:[!0===y&&"true",!1===y&&"false",(null===y||"null"===y)&&"null",!0!==y&&!1!==y&&null!==y&&"null"!==y&&y]}),h&&(0,O.jsx)(l(),{as:(0,S.o_)(w.mW,x),href:"/datasets/[...slug]",passHref:!0,children:(0,O.jsx)(a.Z,{danger:!0,children:"View all"})})]}))}))}))}),[z,k,C,W,P,K,Q,V,T]),$=(0,i.useMemo)((function(){var e;return I?(e=(0,j.Sm)(Q.map(M)),e+=n||Z-y.dN):(e=x,e-=n||Z),e}),[n,x,I,Q]),ee=(0,i.useMemo)((function(){return(0,O.jsx)(d.S_,{estimatedItemSize:Z,height:$,itemCount:null===Q||void 0===Q?void 0:Q.length,itemSize:function(e){return M(Q[e])},outerRef:L,style:{maxHeight:I,pointerEvents:c?"none":null},children:Y})}),[c,$,I,Y,Q]);return(0,O.jsx)("div",E(E({},J()),{},{className:"table sticky",style:{width:_},children:(0,O.jsxs)("div",E(E({},q()),{},{className:"body",children:[(0,O.jsx)("div",{className:"header",ref:D,children:X.map((function(e,n){return(0,i.createElement)("div",E(E({},e.getHeaderGroupProps()),{},{className:"tr",key:"".concat(e.id,"_").concat(n)}),e.headers.map((function(e,n){var t,r=n<=P-1,o=e.getHeaderProps(),a=E({},o.style);return r?(a.fontFamily=b.Vp,a.left=0,a.position="sticky",a.textAlign="center",a.width=W[n],a.minWidth=W[n]):N?t=N(e,n-P,{width:G.width}):(t=e.render("Header"),a.color=(R||p.Z).content.default,a.padding=1*H.iI,a.minWidth=G.width),(0,i.createElement)("div",E(E({},o),{},{className:"th",key:e.id,style:a,title:r?"Row number":void 0}),t)})))}))}),ee]}))}))}var P=function(e){var n=e.columnHeaderHeight,t=e.columns,r=e.disableScrolling,o=e.height,a=e.index,c=e.invalidValues,l=e.maxHeight,u=e.noBorderBottom,d=e.noBorderLeft,s=e.noBorderRight,h=e.noBorderTop,m=e.previewIndexes,f=e.renderColumnHeader,p=e.rows,g=e.width,v=(0,i.useMemo)((function(){return null!==a&&void 0!==a&&a.length&&Array.isArray(a[0])?a[0].length:1}),[a]),b=(0,i.useMemo)((function(){return(0,j.w6)(v).map((function(e,n){return{Header:(0,j.w6)(n+1).map((function(){return" "})).join(" "),accessor:function(e,n){return n},sticky:"left"}})).concat(null===t||void 0===t?void 0:t.map((function(e){return{Header:String(e),accessor:String(e)}})))}),[t,v]),y=(0,i.useMemo)((function(){return(0,O.jsx)(I,{columnHeaderHeight:n,columns:b,data:p,disableScrolling:r,height:o,index:a,invalidValues:c,maxHeight:l,numberOfIndexes:v,previewIndexes:m,renderColumnHeader:f,width:g})}),[n,b,p,r,o,a,c,l,v,m,f,g]);return(0,O.jsx)(C,{columnHeaderHeight:n,disableScrolling:r,height:o,maxHeight:l?l+37:l,noBorderBottom:u,noBorderLeft:d,noBorderRight:s,noBorderTop:h,children:y})}},48888:function(e,n,t){t.d(n,{AE:function(){return o},H3:function(){return i},mW:function(){return a},oE:function(){return c},yg:function(){return r}});var r="tabs[]",o="show_columns",i="column",a="Reports",c="Visualizations"},40489:function(e,n,t){t.d(n,{o_:function(){return l}});var r=t(75582),o=t(34376),i=t(48888);t(82684),t(12691),t(71180),t(58036),t(97618),t(55485),t(48670),t(38276),t(30160),t(72473),t(28598);var a,c=t(69419);!function(e){e.DATASETS="datasets",e.DATASET_DETAIL="dataset_detail",e.COLUMNS="features",e.COLUMN_DETAIL="feature_detail",e.EXPORT="export"}(a||(a={}));var l=function(e,n){var t=(0,o.useRouter)().query.slug,l=void 0===t?[]:t,u=(0,r.Z)(l,1)[0],d=(0,c.iV)(),s=d.show_columns,h=d.column,m="/".concat(a.DATASETS,"/").concat(u),f="".concat(i.H3,"=").concat(h||n),p="".concat(i.yg,"=").concat(e,"&").concat(f,"&").concat(i.AE,"=").concat(s||0);return"".concat(m,"?").concat(p)}}}]);