(* ::Package:: *)

DefineSinfTable["https://www3.risc.jku.at/research/combinat/software/HarmonicSums/RelTabSinfH1.m"];
DefineCSinfTable["https://www3.risc.jku.at/research/combinat/software/HarmonicSums/RelTabCSinf.m"];

InvMellinRules={
	InvMellinProblem[1/(1.` +1.` n),n,x] -> 1,
	InvMellinProblem[1/(2.` +1.` n),n,x] -> x,
	InvMellinProblem[1/(3.` +1.` n),n,x] -> x^2,
	InvMellinProblem[1/(4.` +1.` n),n,x] -> x^3,
	InvMellinProblem[1/(5.` +1.` n),n,x] -> x^4,
	InvMellinProblem[S[1,n]/(1.` +1.` n),n,x] -> -Log[1-x]+Log[x],
	InvMellinProblem[S[2,n]/(1.` +1.` n),n,x] -> Log[1-x] Log[x]-Log[x]^2/2+PolyLog[2,x],
	InvMellinProblem[S[3,n]/(1.` +1.` n),n,x] -> 1/3! Log[x]^3 + H[1,0,0,x],
	InvMellinProblem[S[4,n]/(1.` +1.` n),n,x] -> - 1/4! Log[x]^4 - H[1,0,0,0,x],
	InvMellinProblem[S[1,n]/(2.` +1.` n),n,x] -> -1+x-x Log[1-x]+x Log[x],
	InvMellinProblem[1/(1.` +1.` n)^7,n,x] -> Log[x]^6/6!,
	InvMellinProblem[1/(1.` +1.` n)^6,n,x] -> -Log[x]^5/5!,
	InvMellinProblem[1/(1.` +1.` n)^5,n,x] -> Log[x]^4/4!,
	InvMellinProblem[1/(1.` +1.` n)^4,n,x] -> -Log[x]^3/3!,
	InvMellinProblem[1/(1.` +1.` n)^3,n,x] -> Log[x]^2/2,
	InvMellinProblem[1/(1.` +1.` n)^2,n,x] -> -Log[x],
	InvMellinProblem[1/(3.` +1.` n)^2,n,x] -> -x^2 Log[x],
	InvMellinProblem[S[1,1,n]/(1.` +1.` n)^2,n,x] -> 1/6 \[Pi]^2 Log[x]-PolyLog[3,x]+1/2 (-(1/3) Log[1-x] (\[Pi]^2-3 Log[1-x] Log[x])+2 Log[1-x] PolyLog[2,x]+2 PolyLog[3,1-x]-2 Zeta[3])+2 Zeta[3],
	InvMellinProblem[S[1,n]/(1.` +1.` n)^2,n,x] -> -(1/2) Log[x]^2-PolyLog[2,x]+Zeta[2],
	InvMellinProblem[S[2,n]/(1.` +1.` n)^2,n,x] -> 1/3! Log[x]^3+H[0,1,0,x]+2 Zeta[3],
	InvMellinProblem[S[1,n]/(2.` +1.` n)^3,n,x] -> -1+x-x z3+x 1/3! Log[x]^3 +x H[0,0,1,x]-x Log[x]-x z2 Log[x]+1/2 x Log[x]^2,
	InvMellinProblem[S[1,n]/(3.` +1.` n),n,x] -> -(1/2)-x+(3 x^2)/2-x^2 Log[1-x] +x^2 Log[x],
	InvMellinProblem[1/(2.`+ 1.`n)^5,n,x] -> x 1/4! Log[x]^4,
	InvMellinProblem[1/(2.`+ 1.`n)^4,n,x] -> -x 1/3! Log[x]^3,
	InvMellinProblem[1/(2.`+ 1.`n)^3,n,x] -> 1/2 x Log[x]^2,
	InvMellinProblem[1/(2.`+ 1.`n)^2,n,x] -> - x Log[x],
	InvMellinProblem[S[1,n]/(1.` +1.` n)^3,n,x] ->  1/3! Log[x]^3 +H[0,0,1,x] -Log[x] Zeta[2]-Zeta[3],
	InvMellinProblem[S[1,n]/(1.` +1.` n)^4,n,x] ->  (2 Zeta[2]^2)/5-1/4! Log[x]^4 +Zeta[3] Log[x]+1/2 Zeta[2] Log[x]^2-PolyLog[4,x],
	InvMellinProblem[1/(2 + n)^2,n,x] -> -x Log[x],
	InvMellinProblem[S[1,n]/(2.` +1.` n)^2,n,x] -> -1+x+x z2-x Log[x]-1/2 x Log[x]^2-x PolyLog[2,x],
	InvMellinProblem[(1/2 S[1,n]^2+1/2 S[2,n])/(1.` +1.` n),n,x] -> -Zeta[2]+H[1,1,x]+PolyLog[2,x],
	InvMellinProblem[Lm16[1+n],n,x] -> Log[1-x]^6,
	InvMellinProblem[Lm15[1+n],n,x] -> Log[1-x]^5,
	InvMellinProblem[Lm14[1+n],n,x] -> Log[1-x]^4,
	InvMellinProblem[Lm13[1+n],n,x] -> Log[1-x]^3,
	InvMellinProblem[Lm12[1+n],n,x] -> Log[1-x]^2,
	InvMellinProblem[Lm11[1+n],n,x] -> Log[1-x],
	InvMellinProblem[Lm14m1[1+n],n,x] -> (1-x)Log[1-x]^4,
	InvMellinProblem[Lm13m1[1+n],n,x] -> (1-x)Log[1-x]^3,
	InvMellinProblem[Lm12m1[1+n],n,x] -> (1-x)Log[1-x]^2,
	InvMellinProblem[Lm11m1[1+n],n,x] -> (1-x)Log[1-x],
	InvMellinProblem[Lm14m2[1+n],n,x] -> (1-x)^2Log[1-x]^4,
	InvMellinProblem[Lm13m2[1+n],n,x] -> (1-x)^2Log[1-x]^3,
	InvMellinProblem[Lm12m2[1+n],n,x] -> (1-x)^2Log[1-x]^2,
	InvMellinProblem[Lm11m2[1+n],n,x] -> (1-x)^2Log[1-x],
	InvMellinProblem[Lm14[2+n],n,x] -> 24 x H[1,1,1,1,x],
	InvMellinProblem[Lm13[2+n],n,x] -> -6 x H[1,1,1,x],
	InvMellinProblem[Lm12[2+n],n,x] -> 2 x H[1,1,x],
	InvMellinProblem[Lm11[2+n],n,x] -> x Log[1-x],
	InvMellinProblem[S12m1[1+n],n,x] -> Log[1-x]^2-Log[1-x] Log[x]-PolyLog[2,x],
	InvMellinProblem[S13m1[1+n],n,x] -> -(1/2) \[Pi]^2 Log[1-x]-Log[1-x]^3+3 Log[1-x]^2 Log[x]-1/2 Log[1-x] Log[x]^2+(3 Log[1-x]-Log[x]) PolyLog[2,x]+3 PolyLog[3,1-x]+PolyLog[3,x]+3 Log[1-x] Zeta[2]-3 Zeta[3],
	InvMellinProblem[S11m2[1+n],n,x] -> Zeta[2]-PolyLog[2,x],
	InvMellinProblem[S12m2[1+n],n,x] -> -(1/3) \[Pi]^2 Log[1-x]+Log[1-x]^2 Log[x]+(2 Log[1-x]-Log[x]) PolyLog[2,x]+2 PolyLog[3,1-x]+2 PolyLog[3,x]-2 Zeta[3],
	InvMellinProblem[S13m2[1+n],n,x] -> -(\[Pi]^4/15)-1/2 \[Pi]^2 Log[1-x] Log[x]+Log[1-x]^3 Log[x]+3/2 Log[1-x]^2 Log[x]^2+3 Log[1-x]^2 PolyLog[2,1-x]+(3 z2+3 Log[1-x] Log[x]-Log[x]^2/2) PolyLog[2,x]-6 Log[1-x] PolyLog[3,1-x]+3 Log[x] PolyLog[3,1-x]+2 Log[x] PolyLog[3,x]+6 PolyLog[4,1-x]-3 PolyLog[4,x]+6 PolyLog[2,2,x]-3 Log[x] Zeta[3],
	InvMellinProblem[SuppLog[1+n],n,x] -> -1+1/x,
	InvMellinProblem[S[2,n]/(2.` +1.` n),n,x] -> 1-x+Log[x]-1/2 x Log[x]^2-x (-Log[1-x] Log[x]-PolyLog[2,x]),
	InvMellinProblem[S[1,1,1,n]/(1.` +1.` n)^2,n,x] -> -(\[Pi]^4/90)+z2^2/2+z3 Log[x]+1/6 Log[1-x]^3 Log[x]+1/2 Log[1-x]^2 PolyLog[2,1-x]-Log[1-x] PolyLog[3,1-x]+PolyLog[4,1-x]-PolyLog[2,2,x],
	InvMellinProblem[S[1,1,n]/(1.` +1.` n),n,x] -> -z2+1/2 Log[1-x]^2+PolyLog[2,x],
	InvMellinProblem[S[3,n]/(3.` +1.` n),n,x] -> 1/24 (-3-24 x+27 x^2-6 Log[x]-24 x Log[x]-6 Log[x]^2-12 x Log[x]^2+4 x^2 Log[x]^3-12 x^2 (Log[1-x] Log[x]^2+2 Log[x] PolyLog[2,x]-2 PolyLog[3,x])),
	InvMellinProblem[S[1,n]/(4.` +1.` n),n,x] -> 1/6 (-2-3 x-6 x^2+11 x^3-6 x^3 Log[1-x]+6 x^3 Log[x]),
	InvMellinProblem[S[3,n]/(2.` +1.` n)^3,n,x] -> -6+6 x-6 x z5-3 Log[x]-3 x Log[x]-6/5 x z2^2 Log[x]-Log[x]^2/2+1/2 x Log[x]^2+1/120 x Log[x]^5+1/2 x Log[x]^2 PolyLog[3,x]-3 x Log[x] PolyLog[4,x]+6 x PolyLog[5,x],
	InvMellinProblem[S[3,n]/(2.` +1.` n)^2,n,x] -> -3+3 x+(6 x z2^2)/5-2 Log[x]-x Log[x]-Log[x]^2/2-1/24 x Log[x]^4-1/2 x Log[x]^2 PolyLog[2,x]+2 x Log[x] PolyLog[3,x]-3 x PolyLog[4,x],
	InvMellinProblem[S[1,1,1,n]/(1.` +1.` n),n,x] -> 1/6 (-6 z3+\[Pi]^2 Log[1-x]-Log[1-x]^3-3 Log[1-x]^2 Log[x]-6 Log[1-x] PolyLog[2,x]-6 PolyLog[3,1-x]+6 Zeta[3]),
    InvMellinProblem[S[1,1,n]/(2.` +1.` n),n,x] -> -(-1+x) Log[1-x]+1/2 x Log[1-x]^2+x (-z2+Log[x])+x PolyLog[2,x],
    InvMellinProblem[S[3,n]/(2.` +1.` n),n,x] -> -1+x-Log[x]-Log[x]^2/2+1/6 x Log[x]^3+x (-(1/2) Log[1-x] Log[x]^2-Log[x] PolyLog[2,x]+PolyLog[3,x]),
    InvMellinProblem[S[1,1,n]/(1.` +1.` n)^3,n,x] -> -(z2^2/2)-2 z3 Log[x]-1/2 z2 Log[x]^2+PolyLog[4,x]+PolyLog[2,2,x],
    InvMellinProblem[S[1,1,1,1,n]/(1.` +1.` n),n,x] -> \[Pi]^4/90-(2 z2^2)/5+1/24 Log[1-x]^4-1/6 Log[1-x]^3 Log[x]-1/2 Log[1-x]^2 PolyLog[2,1-x]+Log[1-x] PolyLog[3,1-x]-PolyLog[4,1-x],
    InvMellinProblem[S[1,1,1,n]/(2.` +1.` n),n,x] -> 1/6 (-\[Pi]^2 x+\[Pi]^2 x Log[1-x]-3 Log[1-x]^2+3 x Log[1-x]^2-x Log[1-x]^3-3 x Log[1-x]^2 Log[x]-6 x (-1+Log[1-x]) PolyLog[2,x]-6 x PolyLog[3,1-x]),
    InvMellinProblem[S[1,1,n]/(2.` +1.` n)^2,n,x] -> 1/6 (\[Pi]^2 x+6 Log[1-x]-6 x Log[1-x]-\[Pi]^2 x Log[1-x]+6 x Log[x]+\[Pi]^2 x Log[x]+3 x Log[1-x]^2 Log[x]-3 x Log[x]^2+6 x (-1+Log[1-x]) PolyLog[2,x]+6 x PolyLog[3,1-x]-6 x PolyLog[3,x]+6 x Zeta[3]),
    InvMellinProblem[1/(12.` +7.` n+1.` n^2),n,x]->x^2-x^3,
    InvMellinProblem[1/(6.` +5.` n+1.` n^2),n,x] -> x-x^2,
    InvMellinProblem[n/(6.` +5.` n+1.` n^2),n,x]->-2 x+3 x^2,
    InvMellinProblem[1/(2.` +3.` n+1.` n^2),n,x]->1-x,
    InvMellinProblem[1/(8.` +6.` n+1.` n^2),n,x]->x/2-x^3/2
};


Ht[-1,0,x] := H[-1,0,x] + Zeta[2]/2;
Ht[1,0,x] := H[1,0,x] + Zeta[2];
Ht[1,0,0,x] := H[1,0,0,x] - Zeta[3];
Ht[0,-1,0,x] := H[0,-1,0,x] + H[0,x] Zeta[2]/2 + 3 Zeta[3]/2;
Ht[0,1,0,x] := H[0,1,0,x] + H[0,x] Zeta[2] + 2 Zeta[3];


HarmonicRules={
 Zeta[n__, a__] -> PolyGamma[n-1, a]/Factorial[n-1](-1)^(-n),
 PolyGamma[0, n__] -> S[1, - 1 + n] - EulerGamma,
 PolyGamma[1, n__] -> - S[2, - 1 + n] + Zeta[2],
 PolyGamma[2, n__] -> 2 S[3, - 1 + n] - 2 Zeta[3],
 PolyGamma[3, n__] -> - 6 S[4, - 1 + n] + 6 Zeta[4],
 PolyGamma[4, n__] -> 24 S[5, - 1 + n] - 24 Zeta[5],
 PolyGamma[5, n__] -> - 120 S[6, - 1 + n] + 120 Zeta[6],
 PolyGamma[6, n__] -> 720 S[7, - 1 + n] - 720 Zeta[7]
};

ZetaRules = {z2 -> Zeta[2], z3 -> Zeta[3], z4 -> Zeta[4], z5 -> Zeta[5], z6 -> Zeta[6], z7 -> Zeta[7]};

hrep={
	H[0,0,0,x]  -> 1/3! Log[x]^3,
	H[0,0,0,0,x]  -> 1/4! Log[x]^4,
	H[0,0,0,0,0,x]  -> 1/5! Log[x]^5,
	H[0,0,0,0,0,0,x]  -> 1/6! Log[x]^6,
	H[0,1,1,x] -> -(1/2) (-(1/3) Log[1-x] (\[Pi]^2-3 Log[1-x] Log[x])+2 Log[1-x] PolyLog[2,x]+2 PolyLog[3,1-x]-2 Zeta[3]),
	H[0,1,0,x] -> Log[x] PolyLog[2,x]-2 PolyLog[3,x],
	H[0,0,1,x]-> PolyLog[3,x],
	H[1,x] -> - Log[1-x],
	H[1,1,x] ->  1/2 Log[1-x]^2,
	H[0,x]->Log[x],
	H[2,x]->Log[2]-Log[2-x],
	H[0,-1,x]->-PolyLog[2,-x],
	H[0,0,x]->Log[x]^2/2,
	H[0,1,x]->PolyLog[2,x],
	H[0,2,x]->\[Pi]^2/6+Log[-(2/(-2+x))] Log[x/2]-PolyLog[2,1-x/2],
	H[1,0,x]->-Log[1-x] Log[x]-PolyLog[2,x],
	H[2,0,x]->1/6 (\[Pi]^2+3 Log[2]^2-3 Log[2-x]^2+6 I \[Pi] Log[1-x/2]-6 PolyLog[2,-(2/(-2+x))]),
	H[2,2,x]->1/2 (Log[2]-Log[2-x])^2,
	H[0,0,2,x]->PolyLog[3,x/2],
	H[0,2,0,x]->Log[x] PolyLog[2,x/2]-2 PolyLog[3,x/2],
	H[0,2,1,x]->1/12 \[Pi]^2 Log[2]-1/2 I \[Pi] Log[2]^2+Log[2]^2 Log[1-x]+1/2 Log[2] Log[1-x]^2-Log[2] Log[1-x] Log[2-x]+1/2 I \[Pi] Log[2-x]^2+1/12 \[Pi]^2 Log[x]+2 Log[1-x] Log[2-x] Log[x]+Log[2-2 x] PolyLog[2,1-x]+Log[2-x] PolyLog[2,2-x]-Log[2] PolyLog[2,1-x/2]+Log[2-x] PolyLog[2,1-x/2]+Log[2] PolyLog[2,(-2+x)/(2 (-1+x))]+Log[1-x] PolyLog[2,(-2+x)/(2 (-1+x))]-Log[2-x] PolyLog[2,(-2+x)/(2 (-1+x))]-Log[2] PolyLog[2,(-2+x)/(-1+x)]-Log[1-x] PolyLog[2,(-2+x)/(-1+x)]+Log[2-x] PolyLog[2,(-2+x)/(-1+x)]+Log[x] PolyLog[2,-1+x]+Log[x] PolyLog[2,x]+Log[2-x] PolyLog[2,-(x/(-2+x))]-Log[x] PolyLog[2,-(x/(-2+x))]-Log[2-x] PolyLog[2,x/(-2+x)]+Log[x] PolyLog[2,x/(-2+x)]-PolyLog[3,1-x]-PolyLog[3,2-x]-PolyLog[3,1-x/2]+PolyLog[3,(-2+x)/(2 (-1+x))]-PolyLog[3,(-2+x)/(-1+x)]-PolyLog[3,x]+PolyLog[3,-(x/(-2+x))]-PolyLog[3,x/(-2+x)]+(11 Zeta[3])/4,
	H[0,2,2,x]->1/2 (-((I \[Pi]^3)/3)+2 I \[Pi] Log[2]^2+Log[2]^3-1/3 \[Pi]^2 Log[16]-2 I \[Pi] Log[2] Log[2-x]+\[Pi]^2 Log[2/x]-Log[2-x]^2 Log[2/x]+\[Pi]^2 Log[x]-2 I \[Pi] Log[2] Log[x]+2 I \[Pi] Log[2-x] Log[x]-Log[2-x]^2 Log[x]+2 Log[2-x] Log[1-x/2] Log[x]+2 Log[1-x/2] Log[-(2/(-2+x))] Log[x]+Log[-(2/(-2+x))]^2 Log[x]+2 (I \[Pi]+Log[2-x]) PolyLog[2,1-x/2]+(2 I \[Pi]+Log[4]) PolyLog[2,x/2]-2 PolyLog[3,1-x/2]+2 Zeta[3]),
	H[2,0,0,x]->1/2 Log[-(2/(-2+x))] Log[x]^2-Log[x] PolyLog[2,x/2]+PolyLog[3,x/2],H[2,0,1,x]->(I \[Pi]^3)/6-1/12 \[Pi]^2 Log[2]+1/2 I \[Pi] Log[2]^2-Log[2]^2 Log[1-x]-1/2 Log[2] Log[1-x]^2+Log[2] Log[1-x] Log[2-x]-I \[Pi] Log[1-x] Log[x]-Log[1-x] Log[2-x] Log[x]-(I \[Pi]+Log[2-2 x]) PolyLog[2,1-x]-Log[1-x/2] PolyLog[2,1-x/2]-Log[2] PolyLog[2,(-2+x)/(2 (-1+x))]-Log[1-x] PolyLog[2,(-2+x)/(2 (-1+x))]+Log[2-x] PolyLog[2,(-2+x)/(2 (-1+x))]+Log[2] PolyLog[2,(-2+x)/(-1+x)]+Log[1-x] PolyLog[2,(-2+x)/(-1+x)]-Log[2-x] PolyLog[2,(-2+x)/(-1+x)]-I \[Pi] PolyLog[2,x]-Log[2-x] PolyLog[2,x]+PolyLog[3,1-x]+PolyLog[3,1-x/2]-PolyLog[3,(-2+x)/(2 (-1+x))]+PolyLog[3,(-2+x)/(-1+x)]-(15 Zeta[3])/8,
	H[2,0,2,x]->Log[2]^3-Log[2-x] Log[-(4/(-2+x))] Log[2/x]-Log[2]^2 Log[x]+Log[-(2/(-2+x))] (2 PolyLog[2,1-x/2]+PolyLog[2,x/2])+2 PolyLog[3,1-x/2]-2 Zeta[3],H[2,1,0,x]->(I \[Pi]^3)/6-1/6 \[Pi]^2 Log[2]-1/6 \[Pi]^2 Log[1-x]+\[Pi]^2 Log[2-x]-I \[Pi] Log[1-x] Log[2-x]+3/2 Log[1-x]^2 Log[2-x]+1/6 \[Pi]^2 Log[-(2/(-2+x))]-I \[Pi] Log[2-x] Log[x]-Log[1-x] Log[2-x] Log[x]+1/2 I \[Pi] Log[x]^2+Log[-1+2/x] PolyLog[2,-1+2/x]+Log[2-x] PolyLog[2,1/(1-x)]-I \[Pi] PolyLog[2,2-x]+Log[1-x] PolyLog[2,2-x]-Log[2-x] PolyLog[2,2-x]-I \[Pi] PolyLog[2,-1+x]+Log[1-x] PolyLog[2,-1+x]-Log[2-x] PolyLog[2,(-2+x)/x]+Log[x] PolyLog[2,(-2+x)/x]-Log[x] PolyLog[2,x]-PolyLog[3,-1+2/x]+PolyLog[3,2-x]+PolyLog[3,(-2+x)/x]+PolyLog[3,x]-(7 Zeta[3])/8,H[2,1,1,x]->-(1/2) Log[1-x]^2 Log[2-x]-Log[1-x] PolyLog[2,-1+x]+PolyLog[3,-1+x]+(3 Zeta[3])/4,H[2,2,0,x]->1/6 (-Log[2]^3+Log[2-x]^3-(\[Pi]^2+3 Log[2]^2) Log[1-x/2]-3 I \[Pi] Log[-(2/(-2+x))]^2)-PolyLog[3,-(2/(-2+x))]+Zeta[3],
	H[2,2,1,x]->-(1/12) \[Pi] (-6 I Log[2-x]^2+\[Pi] Log[-(-2+x)^3])+PolyLog[3,2-x]-(7 Zeta[3])/8,
	H[2,2,2,x]->1/6 (Log[2]-Log[2-x])^3,
	H[0,-1,-1,0,x]->-(1/2) (Log[x] Log[1+x]+PolyLog[2,-x])^2+Log[x] (1/2 Log[-x] Log[1+x]^2-1/2 Log[x] Log[1+x]^2+Log[1+x] (Log[x] Log[1+x]+PolyLog[2,-x])+Log[1+x] PolyLog[2,1+x]-PolyLog[3,1+x]+Zeta[3]),
	H[0,-1,0,0,x]->1/2 Log[x]^2 PolyLog[2,-x]+Log[x]^2 (Log[x] Log[1+x]+PolyLog[2,-x])-2 Log[x] (1/2 Log[x] PolyLog[2,-x]+1/2 Log[x] (Log[x] Log[1+x]+PolyLog[2,-x])-PolyLog[3,-x])-3 PolyLog[4,-x],
	H[0,0,-1,0,x]->-(1/2) Log[x]^2 PolyLog[2,-x]-1/2 Log[x]^2 (Log[x] Log[1+x]+PolyLog[2,-x])+Log[x] (1/2 Log[x] PolyLog[2,-x]+1/2 Log[x] (Log[x] Log[1+x]+PolyLog[2,-x])-PolyLog[3,-x])+3 PolyLog[4,-x],
	H[0,0,0,1,x]->PolyLog[4,x],
	H[0,0,1,1,x]->PolyLog[2,2,x],
	H[0,1,0,0,x]->1/2 Log[x]^2 PolyLog[2,x]-2 Log[x] PolyLog[3,x]+3 PolyLog[4,x],
	H[0,1,0,1,x]->1/2 PolyLog[2,x]^2-2 PolyLog[2,2,x],
	H[0,1,1,1,x]->\[Pi]^4/90-1/6 Log[1-x]^3 Log[x]-1/2 Log[1-x]^2 PolyLog[2,1-x]+Log[1-x] PolyLog[3,1-x]-PolyLog[4,1-x],
	H[-1,x] -> Log[1+x],
	H[0,0,-1,x] -> -PolyLog[3,-x],
	H[0,0,0,-1,x] -> -PolyLog[4,-x],
	H[0,0,0,0,1,x] -> PolyLog[5,x],
	H[0,0,0,1,1,x] -> PolyLog[3,2,x]
};

QCDConstantsRules = {
	ca -> 3,
	nc -> 3,
	cf -> 4/3,
	tr -> 1/2,
	d4RA/nr -> 5/2, (*(nc^2+6)(nc^2-1)/48,*)
	d4RR/nr -> 5/36, (*(nc^4 - 6nc^2 + 18)(nc^2-1)/ (96 nc^3),*)
	d4AA/na -> nc^2(nc^2 + 36) / 24,
	d4RA/na -> nc (nc^2 + 6) / 48,
	d4RR/na -> (nc^4 - 6nc^2 + 18) / (96 nc^2),
	caf -> (ca - cf)
};

LargeNcQCDConstantsRules = {
	ca -> 3,
	nc -> 3,
	cf -> 3/2,
	tr -> 1/2,
	d4RA/nr -> nc^4/48,
	d4RR/nr -> nc^3/96,
	d4AA/na -> nc^4/ 24,
	d4RA/na -> nc^3/ 48,
	d4RR/na -> nc^2/ 96
};

BetaRules={
	beta0 -> 11/3 ca \[Minus] 2/3 nf,
	beta1 -> 34/3 ca^2 \[Minus] 10/3 ca nf \[Minus] 2 cf nf,
	beta2 -> (
        2857/54 ca^3
        - 1415/27 ca^2 tr nf
        - 205/9 cf ca tr nf
        + 2 cf^2 tr nf
        + 44/9 cf (tr nf)^2
        + 158/27 ca (tr nf)^2
    )
};


(* Single argument harmonic sum*)
harmonics = {
 S[1, n] -> S1,  S[2, n] -> S2, S[3, n] -> S3, S[4, n] -> S4, S[5, n] -> S5,
 S[1, n/2] -> Sp1p, S[2, n/2] -> Sp2p, S[3, n/2] -> Sp3p, S[4, n/2] -> Sp4p, S[5, n/2] -> Sp5p
};

harmonicminus = {
S[-1,n] -> Sm1, S[-2,n] -> Sm2, S[-3,n] -> Sm3, S[-4,n] -> Sm4, S[-5,n] -> Sm5
};
(* Multiple argument harmonic sum*)
harmonics234 = {
(* 2nd order *)
S[1,1,n] -> 1/2 (S[1,n]^2 + S[2,n]),

(* 3rd order *)
S[1,1,1,n] -> 1/6(S[1,n]^3 + 3 S[1,n]S[2,n] +2 S[3,n]),
S[1,-2,n] -> - Sm21 + S[-3, n] + S[-2, n] S[1,n],
S[1,2,n] -> - S21 + S[3,n] + S[2,n] S[1,n],
S[-2,-1,n] -> Sm2m1,
S[-2,1,n] -> Sm21,
S[2,-1,n] -> S2m1,
S[2,1,n] -> S21,

 (* 4th order *)
 S[-2,-2,n] -> 1/2 (S[-2,n]^2 + S[4,n]),
 S[1,1,-2,n] -> S[-2,1,1,n] + S[-2,n] S[2,n] - S[-2,2,n] - S[-2,n] S[1,1,n] + S[1,n] S[1,-2,n] + S[1,-3,n] - S[1,n] S[-3,n],
 S[1,-3,n] -> - S[-3,1, n] + S[-3,n] S[1,n] + S[-4, n],
 S[1,3,n] -> - S[3,1,n] + S[3,n] S[1,n] + S[4,n],
 S[2,-2, n] -> -S[-2,2, n] + S[-4,n] + S[-2, n] S[2,n],
 S[2,2, n] -> 1/2( S[2,n]^2 + S[4,n]),
 S[-3,1,n] -> Sm31,
 S[-2,2,n] -> Sm22,
 S[3,1,n] -> S31,
 S[-2,1,1,n] -> Sm211,
 S[2,1,1,n] -> S211,
 S[1,1,1,1,n] -> 1/4 S[4,n] +1/8 S[2,n]^2 + 1/3 S[3,n] S[1,n] + 1/4 S[2,n] S[1,n]^2 + 1/24 S[1,n]^4,
 S[1,1,2,n] -> S[2,1,1,n] + 1/2 ((S[1,n](S[1,2,n]-S[2,1,n])) + S[1,3,n]-S[3,1,n]),
 S[1,2,1,n] -> -2 S[2,1,1,n] + S[3,1,n] + S[1,n]S[2,1,n] + S[2,2,n]
};
harmonics5={
 (* 5th order *)
 S[4,1,n] -> S41,
 S[2,1,-2,n] -> S21m2,
 S[2,2,1,n] -> S221,
 S[-2,2,1,n] -> Sm221,
 S[3,1,1,n] -> S311,
 S[2,1,1,1,n] -> S2111,
 S[-2,1,1,1,n] -> Sm2111,
 S[2,3,n] -> S23,
 S[2,-3,n] -> S2m3,
 S[-2,3,n] -> Sm23
};


(*
Mellin tansform of the large N limit logs (1-x)Log[1-x]^k
Integrate[(1-x)Log[1-x]^k x^(n-1),{x,0,1}, Assumptions\[Rule]{n>0}]
*)
Lm11m1[n_Real]:= 1/(1+n)^2-(EulerGamma+PolyGamma[0,1+n])/(1+n)^2-(EulerGamma+PolyGamma[0,1+n])/(n (1+n)^2);
Lm12m1[n_Real]:= - (2/(1+n)^3)-(2 (EulerGamma+PolyGamma[0,1+n]))/(1+n)^2+(EulerGamma+PolyGamma[0,1+n])^2/n-(EulerGamma+PolyGamma[0,1+n])^2/(1+n)+(\[Pi]^2/6-PolyGamma[1,1+n])/n-(\[Pi]^2/6-PolyGamma[1,1+n])/(1+n);
Lm13m1[n_Real]:= -(1/(2 n (1+n)))(-6 EulerGamma^2+2 EulerGamma^3-\[Pi]^2+EulerGamma \[Pi]^2+6 (-1+EulerGamma) PolyGamma[0,2+n]^2+2 PolyGamma[0,2+n]^3+PolyGamma[0,2+n] (-12 EulerGamma+6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,2+n])-6 (-1+EulerGamma) PolyGamma[1,2+n]+2 PolyGamma[2,2+n]+4 Zeta[3]);
Lm14m1[n_Real]:= 1/20 (1/n (20 EulerGamma^4+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4+80 EulerGamma PolyGamma[0,1+n]^3+20 PolyGamma[0,1+n]^4+20 PolyGamma[0,1+n]^2 (6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,1+n])-20 (6 EulerGamma^2+\[Pi]^2) PolyGamma[1,1+n]+60 PolyGamma[1,1+n]^2+80 EulerGamma PolyGamma[2,1+n]-20 PolyGamma[3,1+n]+160 EulerGamma Zeta[3]+40 PolyGamma[0,1+n] (2 EulerGamma^3+EulerGamma \[Pi]^2-6 EulerGamma PolyGamma[1,1+n]+2 PolyGamma[2,1+n]+4 Zeta[3]))-1/(1+n) (20 EulerGamma^4+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4+80 EulerGamma PolyGamma[0,2+n]^3+20 PolyGamma[0,2+n]^4+20 PolyGamma[0,2+n]^2 (6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,2+n])-20 (6 EulerGamma^2+\[Pi]^2) PolyGamma[1,2+n]+60 PolyGamma[1,2+n]^2+80 EulerGamma PolyGamma[2,2+n]-20 PolyGamma[3,2+n]+160 EulerGamma Zeta[3]+40 PolyGamma[0,2+n] (2 EulerGamma^3+EulerGamma \[Pi]^2-6 EulerGamma PolyGamma[1,2+n]+2 PolyGamma[2,2+n]+4 Zeta[3])));

(*Integrate[(1-x)^2Log[1-x]^k x^(n-1),{x,0,1}, Assumptions\[Rule]{n>0}]*)
Lm11m2[n_Real]:= (5+3 n-(2 (1+n) (2+n) (EulerGamma+PolyGamma[0,1+n]))/n)/((1+n)^2 (2+n)^2);
Lm12m2[n_Real]:= (6+6 (-3+EulerGamma) EulerGamma+\[Pi]^2+6 (-3+2 EulerGamma+(3+2 n)/(2+3 n+n^2)+PolyGamma[0,1+n]) PolyGamma[0,3+n]-6 PolyGamma[1,3+n])/(3 n (1+n) (2+n));
Lm13m2[n_Real]:= (-(1/(2 n (1+n) (2+n)))(12 EulerGamma-18 EulerGamma^2+4 EulerGamma^3-3 \[Pi]^2+2 EulerGamma \[Pi]^2+6 (-3+2 EulerGamma) PolyGamma[0,3+n]^2+4 PolyGamma[0,3+n]^3+2 PolyGamma[0,3+n] (6-18 EulerGamma+6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,3+n])-6 (-3+2 EulerGamma) PolyGamma[1,3+n]+4 PolyGamma[2,3+n]+8 Zeta[3]));
Lm14m2[n_Real]:= (1/(10 Gamma[3+n]) Gamma[n] (120 EulerGamma^2-120 EulerGamma^3+20 EulerGamma^4+20 \[Pi]^2-60 EulerGamma \[Pi]^2+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4+40 (-3+2 EulerGamma) PolyGamma[0,3+n]^3+20 PolyGamma[0,3+n]^4+20 PolyGamma[0,3+n]^2 (6-18 EulerGamma+6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,3+n])-20 (6-18 EulerGamma+6 EulerGamma^2+\[Pi]^2) PolyGamma[1,3+n]+60 PolyGamma[1,3+n]^2-120 PolyGamma[2,3+n]+80 EulerGamma PolyGamma[2,3+n]-20 PolyGamma[3,3+n]-240 Zeta[3]+160 EulerGamma Zeta[3]+20 PolyGamma[0,3+n] (12 EulerGamma-18 EulerGamma^2+4 EulerGamma^3-3 \[Pi]^2+2 EulerGamma \[Pi]^2-6 (-3+2 EulerGamma) PolyGamma[1,3+n]+4 PolyGamma[2,3+n]+8 Zeta[3])));

(*
Mellin tansform of the large N limit logs Log[1-x]^k
Integrate[ Log[1-x]^k x^(n-1),{x,0,1}, Assumptions\[Rule]{n>0}]
*)
Lm11[n_Real] := -((PolyGamma[0,1+n] + EulerGamma)/n) ;
Lm12[n_Real] := 2/n^3+(2 EulerGamma)/n^2+EulerGamma^2/n+\[Pi]^2/(6 n)+(2 PolyGamma[0,n])/n^2+(2 EulerGamma PolyGamma[0,n])/n+PolyGamma[0,n]^2/n-PolyGamma[1,n]/n;
Lm13[n_Real] := -((\[Pi]^2 (PolyGamma[0,1+n] + EulerGamma)+2 (PolyGamma[0,1+n] + EulerGamma)^3-6 (PolyGamma[0,1+n] + EulerGamma) PolyGamma[1,1+n]+2 PolyGamma[2,1+n]+4 Zeta[3])/(2 n));
Lm14[n_Real] := 1/(20 n) (20 EulerGamma^4+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4+80 EulerGamma PolyGamma[0,1+n]^3+20 PolyGamma[0,1+n]^4+20 PolyGamma[0,1+n]^2 (6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,1+n])-20 (6 EulerGamma^2+\[Pi]^2) PolyGamma[1,1+n]+60 PolyGamma[1,1+n]^2+80 EulerGamma PolyGamma[2,1+n]-20 PolyGamma[3,1+n]+160 EulerGamma Zeta[3]+40 PolyGamma[0,1+n] (2 EulerGamma^3+EulerGamma \[Pi]^2-6 EulerGamma PolyGamma[1,1+n]+2 PolyGamma[2,1+n]+4 Zeta[3]));
Lm15[n_Real] := - (1/(12 n))(12 EulerGamma^5+20 EulerGamma^3 \[Pi]^2+9 EulerGamma \[Pi]^4+60 EulerGamma PolyGamma[0,1+n]^4+12 PolyGamma[0,1+n]^5+20 PolyGamma[0,1+n]^3 (6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,1+n])+180 EulerGamma PolyGamma[1,1+n]^2+120 EulerGamma^2 PolyGamma[2,1+n]+20 \[Pi]^2 PolyGamma[2,1+n]-60 EulerGamma PolyGamma[3,1+n]+12 PolyGamma[4,1+n]+240 EulerGamma^2 Zeta[3]+40 \[Pi]^2 Zeta[3]-60 PolyGamma[1,1+n] (2 EulerGamma^3+EulerGamma \[Pi]^2+2 PolyGamma[2,1+n]+4 Zeta[3])+60 PolyGamma[0,1+n]^2 (2 EulerGamma^3+EulerGamma \[Pi]^2-6 EulerGamma PolyGamma[1,1+n]+2 PolyGamma[2,1+n]+4 Zeta[3])+3 PolyGamma[0,1+n] (20 EulerGamma^4+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4-20 (6 EulerGamma^2+\[Pi]^2) PolyGamma[1,1+n]+60 PolyGamma[1,1+n]^2+80 EulerGamma PolyGamma[2,1+n]-20 PolyGamma[3,1+n]+160 EulerGamma Zeta[3])+288 Zeta[5]);
(*  (* should accidentally cancels accordign to 0912.0369 *)
Lm16[n_Real] := 1/(168 n)(168 EulerGamma^6+420 EulerGamma^4 \[Pi]^2+378 EulerGamma^2 \[Pi]^4+61 \[Pi]^6+1008 EulerGamma PolyGamma[0,1+n]^5+168 PolyGamma[0,1+n]^6+420 PolyGamma[0,1+n]^4 (6 EulerGamma^2+\[Pi]^2-6 PolyGamma[1,1+n])+1260 (6 EulerGamma^2+\[Pi]^2) PolyGamma[1,1+n]^2-2520 PolyGamma[1,1+n]^3+3360 EulerGamma^3 PolyGamma[2,1+n]+1680 EulerGamma \[Pi]^2 PolyGamma[2,1+n]+1680 PolyGamma[2,1+n]^2-2520 EulerGamma^2 PolyGamma[3,1+n]-420 \[Pi]^2 PolyGamma[3,1+n]+1008 EulerGamma PolyGamma[4,1+n]-168 PolyGamma[5,1+n]+6720 EulerGamma^3 Zeta[3]+3360 EulerGamma \[Pi]^2 Zeta[3]+6720 PolyGamma[2,1+n] Zeta[3]+6720 Zeta[3]^2+1680 PolyGamma[0,1+n]^3 (2 EulerGamma^3+EulerGamma \[Pi]^2-6 EulerGamma PolyGamma[1,1+n]+2 PolyGamma[2,1+n]+4 Zeta[3])-126 PolyGamma[1,1+n] (20 EulerGamma^4+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4+80 EulerGamma PolyGamma[2,1+n]-20 PolyGamma[3,1+n]+160 EulerGamma Zeta[3])+126 PolyGamma[0,1+n]^2 (20 EulerGamma^4+20 EulerGamma^2 \[Pi]^2+3 \[Pi]^4-20 (6 EulerGamma^2+\[Pi]^2) PolyGamma[1,1+n]+60 PolyGamma[1,1+n]^2+80 EulerGamma PolyGamma[2,1+n]-20 PolyGamma[3,1+n]+160 EulerGamma Zeta[3])+24192 EulerGamma Zeta[5]+84 PolyGamma[0,1+n] (12 EulerGamma^5+20 EulerGamma^3 \[Pi]^2+9 EulerGamma \[Pi]^4+180 EulerGamma PolyGamma[1,1+n]^2+20 (6 EulerGamma^2+\[Pi]^2) PolyGamma[2,1+n]-60 EulerGamma PolyGamma[3,1+n]+12 PolyGamma[4,1+n]+240 EulerGamma^2 Zeta[3]+40 \[Pi]^2 Zeta[3]-60 PolyGamma[1,1+n] (2 EulerGamma^3+EulerGamma \[Pi]^2+2 PolyGamma[2,1+n]+4 Zeta[3])+288 Zeta[5]));
*)
SuppLog[n_Real] := 1/(n - 1) - 1/n;


InverseHarmonicRules={
 S[1, a_] -> PolyGamma[0, a + 1] + EulerGamma,
 S[2, a_] -> - PolyGamma[1, a + 1] + Zeta[2],
 S[3, a_] -> 1/2 PolyGamma[2, a + 1] + Zeta[3],
 S[4, a_] -> - 1/6 PolyGamma[3, a + 1] + Zeta[4],
 S[5, a_] -> 1/24 PolyGamma[4, a + 1] + Zeta[5],
 S[6, a_] -> - 1/120 PolyGamma[5, a + 1] + Zeta[6]
};


LimitRules= {
S[2, Infinity] -> z2,
S[3, Infinity] -> z3,
S[4, Infinity] -> (2 z2^2)/5,
S[5, Infinity] -> z5,
S[2,1,Infinity] -> 2 z3,
S[2,1,1,Infinity] -> (6 z2^2)/5,
S[2,2,Infinity] -> (7 z2^2)/10,
S[3,1,Infinity]-> z2^2/2,
S[-2, Infinity] -> -(z2/2),
S[-3, Infinity] -> -((3 z3)/4),
S[-4, Infinity] -> -((7 z2^2)/20),
S[3,2,\[Infinity]]->3 z2 z3-(9 z5)/2,
S[4,1,\[Infinity]]->-z2 z3+3 z5,
S[3,1,1,\[Infinity]]->z2 z3-z5/2,
S[2,3,\[Infinity]] -> (-2 z2 z3+(11 z5)/2)
};


S13m1[n_Real] := (PolyGamma[0, n + 1] + EulerGamma)^3/n;
S12m1[n_Real] := (PolyGamma[0, n + 1] + EulerGamma)^2/n;
S11m2[n_Real] := (PolyGamma[0, n + 1] + EulerGamma)/n^2;
S13m2[n_Real] := (PolyGamma[0, n + 1] + EulerGamma)^3/n^2;
S12m2[n_Real] := (PolyGamma[0, n + 1] + EulerGamma)^2/n^2;


FastMellin = {
	Log[x]^6 -> 6!/n^7,
	Log[x]^5 -> - (5!/n^6),
	Log[x]^4 -> 4!/n^5,
	Log[x]^2/x  -> 2/(-1+n)^3,
	(1-x) Log[1-x]^4 -> Lm14m1[n],
	(1-x) Log[1-x]^3 -> Lm13m1[n],

	(* Note these 2 subleading terms where not included previously *)
	(1-x)^2 Log[1-x]^3 -> Lm13m2[n],
	(1-x)^2 Log[1-x]^4 -> Lm14m2[n],

	(* parametrized parts *)
	((1-x) Log[x])/x->-(1/(-1+n)^2)+1/n^2,
	(1-x)/x->1/((-1+n) n),
	(1-x)^2 Log[1-x]^2-> Lm12m2[n],
	(1-x) Log[1 -x]^2 -> Lm12m1[n],
	(1-x) Log[1 -x] -> Lm11m1[n],
	(1-x) Log[x] -> -(1/n^2)+1/(1+n)^2,
	Log[x]^3 -> -(3!/n^4),
	Log[x]^2 -> 2!/n^3,
	(1-x)(1+ 2x)-> 1/n-n/(2+3 n+n^2),
	(1-x)x^2 -> 1/(6+5 n+n^2),
	(1-x) x (1+x)->2/(3+4 n+n^2),
	1-x->1/(n+n^2)
};
