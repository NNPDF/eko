(* ::Package:: *)

(* define beta fnc - note: convention is different from AB, but matches with EKO *)
\[Beta]f[k_][a_]:=Sum[-\[Beta][n]*a^(2+n),{n,0,k}]
(*\[Beta]f[2][a]*)

(* spell out the all order expansion with taylor + diff eq. *)
aExpanded[m_,k_]:=Module[{v},
v=Sum[1/n!t^n (D[a[t],{t,n}]/.{t->0}),{n,0,m}];
v=v//.{Derivative[n_][a][u_]:>(D[\[Beta]f[k][a[t]],{t,n-1}]/.{t->u})};
Collect[v,{a0,t},FullSimplify]
]
Print["Alpha_s evolution"]
Series[aExpanded[5,5],{a[0],0,5}]

Module[{s,ss,gg},
(* get alphas series *)
s=Series[aExpanded[4,4],{a[0],0,4+1}];
(*Print@s;*)
(* we can use this to get the gamma series. We need to invert the arguments *)
ss=s/.{t->-tt};
gg=Series[Sum[(ss)^k  g[k-1],{k,1,4}],{a[0],0,4}];
Print["Scheme A transformation"];
Print@Collect[gg,{a[0],tt}];
]
(* define gamma fnc *)
\[Gamma]f[k_][a_]:=Sum[\[Gamma][n]*a^(1+n),{n,0,k}]
Module[{v,m,w,ww},
m=3;
v=Sum[1/(n!)t^n (D[f[t],{t,n}]/.{t->0}),{n,0,m}];
v=v//. {Derivative[n_][f][u_]:>(D[-\[Gamma]f[m+1][a[t]]f[t],{t,n-1}]/.{t->u})};
v=v//. {Derivative[n_][a][u_]:>(D[\[Beta]f[m+1][a[t]],{t,n-1}]/.{t->u})};
v=v/.{f[0]-> 1};
Print["EKO evolution"];
w=Collect[v+O[a[0]]^(m+1),{a[0],t},Simplify];
Print[w];
Print["Scheme B transformation"];
ww = Collect[w/.{t->-t},{a[0],t},Simplify];
Print@ww;
]
(* Ren scale *)
Module[{s,ss,cc},
(* get alphas series *)
s=Series[aExpanded[4,4],{a[0],0,4+1}];
(*Print@s;*)
(* we can use this to get the C series. *)
ss=s/.{t->-tt};
cc=Series[Sum[(ss)^(m+k)C[k],{k,0,3}],{a[0],0,3}];
Print["Ren variation"];
Print@Collect[Normal@cc,{a[0],tt,\[Beta]@_,C@_},FullSimplify];
]
