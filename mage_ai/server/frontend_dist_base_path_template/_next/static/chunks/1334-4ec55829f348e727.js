"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[1334],{27277:function(e,n,t){var r=t(82394),o=t(21831),i=t(82684),u=t(39643),c=t(44688),l=t(28598);function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function d(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){(0,r.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}n.Z=function(e){var n=e.highlightedItemIndexInitial,t=void 0===n?null:n,r=e.itemGroups,a=e.noResultGroups,s=e.onHighlightItemIndexChange,f=e.onMouseEnterItem,p=e.onMouseLeaveItem,v=e.onSelectItem,h=e.renderEmptyState,m=e.searchQuery,g=e.selectedItem,b=e.setItemRefs,y=e.uuid,j=(0,i.useState)(!0),x=j[0],O=j[1],w=(0,i.useMemo)((function(){var e=[],n=r.reduce((function(n,t){var r=t.items.filter((function(e){return!m||function(e,n){return e.searchQueries.filter((function(e){return String(e).toLowerCase().includes(n.toLowerCase())})).length>=1}(e,m)}));return 0===r.length?n:(e.push.apply(e,(0,o.Z)(r)),n.concat(d(d({},t),{},{items:r})))}),[]);return{itemGroups:n,itemsFlattened:e}}),[r,m]),k=w.itemGroups,I=w.itemsFlattened;a&&0===I.length&&(k.push.apply(k,(0,o.Z)(a)),I.push.apply(I,(0,o.Z)(a.reduce((function(e,n){var t=n.items;return e.concat(t)}),[]))));var Z=(0,i.useRef)(null);Z.current=I.map((function(){return(0,i.createRef)()}));var C=(0,i.useState)(t),P=C[0],D=C[1],E=(0,i.useCallback)((function(e){null===s||void 0===s||s(e),D(e)}),[s,D]),S=I[P],M=(0,c.y)(),_=M.registerOnKeyDown,R=M.unregisterOnKeyDown;(0,i.useEffect)((function(){return function(){return R(y)}}),[R,y]),null===_||void 0===_||_(y,(function(e,n,t){var r,o=!0,i=I.length,c=I.findIndex((function(e,r){var o=e.keyboardShortcutValidation;return null===o||void 0===o?void 0:o({keyHistory:t,keyMapping:n},r)})),l=n[u.Gs]&&!n[u.XR]&&!g;return-1!==c?(e.preventDefault(),v(I[c]),O(o),E(c)):(n[u.Uq]||l)&&I[P]?(l&&e.preventDefault(),v(I[P]),O(o),E(P)):(n[u.Bu]?(o=!1,r=null===P?i-1:P-1):n[u.kD]?(o=!1,r=null===P?0:P+1):n[u.vP]&&E(null),"undefined"!==typeof r&&(r>=i?r=0:r<=-1&&(r=i-1),r>=0&&r<=i-1?(E(r),e.preventDefault()):E(null)),void O(o))}),[P,I,g,E,O]),(0,i.useEffect)((function(){null===b||void 0===b||b(Z)}),[Z,I,b]),(0,i.useEffect)((function(){var e=null===P||"undefined"===typeof P||P>=I.length;(null===m||void 0===m?void 0:m.length)>=1&&e&&E(0)}),[P,I,m,E]);var G=(0,i.useCallback)((function(){return O(!0)}),[O]);return(0,i.useEffect)((function(){return window.addEventListener("mousemove",G),function(){window.removeEventListener("mousemove",G)}}),[G]),0===k.length&&h?h():(0,l.jsx)(l.Fragment,{children:k.map((function(e,n){var t=e.items,r=e.renderItem,o=e.renderGroupHeader,i=e.uuid,u=n>=1?k.slice(0,n).reduce((function(e,n){return e+n.items.length}),0):0,c=t.map((function(e,n){var t=e.itemObject,o=e.value,i=o===(null===S||void 0===S?void 0:S.value),c=n+u,a=(null===t||void 0===t?void 0:t.id)||(null===t||void 0===t?void 0:t.uuid);return(0,l.jsx)("div",{id:"item-".concat(o,"-").concat(a),onMouseMove:function(){return x&&E(c)},ref:Z.current[c],children:r(e,{highlighted:i,onClick:function(){return v(e)},onMouseEnter:function(){return null===f||void 0===f?void 0:f(e)},onMouseLeave:function(){return null===p||void 0===p?void 0:p(e)}},n,c)},"item-".concat(o,"-").concat(a))}));return c.length>=1&&(0,l.jsxs)("div",{children:[null===o||void 0===o?void 0:o(),c]},i||"group-uuid-".concat(n))}))})}},81334:function(e,n,t){t.d(n,{Z:function(){return I}});var r=t(82394),o=t(82684),i=t(27277),u=t(31882),c=t(38276),l=t(48381),a=t(30160),d=t(17488),s=t(38626),f=t(44897),p=t(42631),v=t(47041),h=t(70515),m=s.default.div.withConfig({displayName:"indexstyle__DropdownStyle",componentId:"sc-suwkha-0"})([""," border-radius:","px;max-height:","px;overflow:auto;position:absolute;width:100%;z-index:1;"," ",""],v.w5,p.BG,40*h.iI,(function(e){return"\n    background-color: ".concat((e.theme.background||f.Z.background).popup,";\n    box-shadow: ").concat((e.theme.shadow||f.Z.shadow).popup,";\n  ")}),(function(e){return e.topOffset&&"\n    top: ".concat(e.topOffset-.5*h.iI,"px;\n  ")})),g=s.default.div.withConfig({displayName:"indexstyle__RowStyle",componentId:"sc-suwkha-1"})(["padding:","px;position:relative;z-index:2;&:hover{cursor:pointer;}",""],.5*h.iI,(function(e){return e.highlighted&&"\n    background-color: ".concat((e.theme.interactive||f.Z.interactive).hoverBackground,";\n  ")})),b=t(39643),y=t(3314),j=t(86735),x=t(44688),O=t(28598);function w(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function k(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?w(Object(t),!0).forEach((function(n){(0,r.Z)(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):w(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var I=function(e){var n,t=e.removeTag,r=e.selectTag,s=e.selectedTags,f=void 0===s?[]:s,p=e.tags,v=void 0===p?[]:p,h=e.uuid,w=(0,o.useRef)(null),I=(0,o.useState)(!1),Z=I[0],C=I[1],P=(0,o.useState)(null),D=P[0],E=P[1],S=(0,o.useMemo)((function(){return(0,j.YC)(v||[],"uuid")}),[v]),M=(0,o.useMemo)((function(){return null===S||void 0===S?void 0:S.map((function(e){return{itemObject:e,searchQueries:[e.uuid],value:e.uuid}}))}),[S]),_=(0,o.useMemo)((function(){return(null===D||void 0===D?void 0:D.length)>=1?M.concat({itemObject:{uuid:D},searchQueries:[D],value:"Add tag: ".concat(D)}):M}),[M,D]),R=(0,x.y)(),G=R.registerOnKeyDown,B=R.unregisterOnKeyDown;return(0,o.useEffect)((function(){return function(){return B(h)}}),[B,h]),null===G||void 0===G||G(h,(function(e,n){var t;Z&&n[b.vP]&&(C(!1),null===w||void 0===w||null===(t=w.current)||void 0===t||t.blur())}),[Z,w]),(0,O.jsxs)(O.Fragment,{children:[(0,O.jsx)(l.Z,{onClickTag:t,tags:f}),(0,O.jsxs)(c.Z,{mt:1,style:{position:"relative"},children:[(0,O.jsx)(d.Z,{onBlur:function(){return setTimeout((function(){return C(!1)}),150)},onChange:function(e){return E(e.target.value)},onFocus:function(){return C(!0)},ref:w,value:D||""}),(0,O.jsx)(m,{topOffset:null===w||void 0===w||null===(n=w.current)||void 0===n?void 0:n.getBoundingClientRect().height,children:(0,O.jsx)(i.Z,{itemGroups:[{items:Z?_:[],renderItem:function(e,n){var t=e.value;return(0,O.jsx)(g,k(k({},n),{},{onClick:function(e){var t;(0,y.j)(e),null===n||void 0===n||null===(t=n.onClick)||void 0===t||t.call(n,e)},children:(0,O.jsx)(u.Z,{small:!0,children:(0,O.jsx)(a.ZP,{children:t})})}))}}],onSelectItem:function(e){var n=e.itemObject;null===r||void 0===r||r(n),E(null)},searchQuery:D,uuid:h})})]})]})}},48381:function(e,n,t){var r=t(82684),o=t(31882),i=t(55485),u=t(30160),c=t(86735),l=t(28598);n.Z=function(e){var n=e.onClickTag,t=e.tags,a=void 0===t?[]:t,d=(0,r.useMemo)((function(){return(null===a||void 0===a?void 0:a.length)||0}),[a]),s=(0,r.useMemo)((function(){return(0,c.YC)(a||[],"uuid")}),[a]);return(0,l.jsx)(i.ZP,{alignItems:"center",flexWrap:"wrap",children:null===s||void 0===s?void 0:s.reduce((function(e,t){return e.push((0,l.jsx)("div",{style:{marginBottom:2,marginRight:d>=2?4:0,marginTop:2},children:(0,l.jsx)(o.Z,{onClick:n?function(){return n(t)}:null,small:!0,children:(0,l.jsx)(u.ZP,{children:t.uuid})})},"tag-".concat(t.uuid))),e}),[])})}},31882:function(e,n,t){var r=t(38626),o=t(71180),i=t(55485),u=t(38276),c=t(30160),l=t(44897),a=t(72473),d=t(70515),s=t(61896),f=t(28598),p=r.default.div.withConfig({displayName:"Chip__ChipStyle",componentId:"sc-1ok73g-0"})(["display:inline-block;"," "," "," "," ",""],(function(e){return!e.primary&&"\n    background-color: ".concat((e.theme.background||l.Z.background).tag,";\n  ")}),(function(e){return e.primary&&"\n    background-color: ".concat((e.theme.chart||l.Z.chart).primary,";\n  ")}),(function(e){return!e.small&&"\n    border-radius: ".concat((d.iI+s.Al)/2,"px;\n    height: ").concat(1.5*d.iI+s.Al,"px;\n    padding: ").concat(d.iI/1.5,"px ").concat(1.25*d.iI,"px;\n  ")}),(function(e){return e.small&&"\n    border-radius: ".concat((d.iI/2+s.Al)/2,"px;\n    height: ").concat(s.Al+d.iI/2+2,"px;\n    padding: ").concat(d.iI/4,"px ").concat(d.iI,"px;\n  ")}),(function(e){return e.border&&"\n    border: 1px solid ".concat((e.theme.content||l.Z.content).muted,";\n  ")}));n.Z=function(e){var n=e.border,t=e.children,r=e.disabled,l=e.label,s=e.onClick,v=e.primary,h=e.small;return(0,f.jsx)(p,{border:n,primary:v,small:h,children:(0,f.jsx)(o.Z,{basic:!0,disabled:r,noBackground:!0,noPadding:!0,onClick:s,transparent:!0,children:(0,f.jsxs)(i.ZP,{alignItems:"center",children:[t,l&&(0,f.jsx)(c.ZP,{small:h,children:l}),!r&&s&&(0,f.jsx)(u.Z,{ml:1,children:(0,f.jsx)(a.x8,{default:v,muted:!v,size:h?d.iI:1.25*d.iI})})]})})})}}}]);