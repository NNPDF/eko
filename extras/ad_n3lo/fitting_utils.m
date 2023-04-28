(* ::Package:: *)

AdFitSinglet[moments_, nfmin_, nfmax_, asyN0_, asyN1_, asyNinf_, FitBasis_, print_: False]:=Module[{Nmin, momentslist, nvals},
	 (* Fit the available mellin moments separately for each nf part *)
	 Nmin =2;
	 If[ Length[moments] == 4,
	     nvals = Range[Nmin, Nmin-1+2 Length[moments],2];
	     tmp = moments,
	     nvals = Transpose[moments][[1]];
	     tmp = Transpose[moments][[2]];
	 ];
	 momentslist = MapThread[#1 - asyN0 - asyN1 - asyNinf //.n -> #2 &, {tmp, nvals}];
	 momentslist = momentslist //. QCDConstantsRules /. ZetaRules // N;
	 flist = MapThread[FitNf[momentslist, #, nvals, FitBasis /. InverseHarmonicRules, print]&, {Range[nfmin,nfmax]}];
	 nf^Range[nfmin, nfmax].flist
];

FitNf[moments_, nfpart_, nvals_, FitBasis_, print_]:=Module[{DataTable,f, Nmin, Nmax, title},
	(* Fit the available mellin moments for a given nf *)
	DataTable = MapThread[{#1,Coefficient[#2,nf, nfpart]}&,{nvals, moments}];

	f = Fit[DataTable, FitBasis[[nfpart+1]], n];
	(* Plot the fitted part *)
	Nmin = 0;
	Nmax = nvals[[-1]]+2;
	title = StringForm["Prop to nf^``.", nfpart];
	If[print,Print[PlotFittedTable[DataTable, f,Nmin,Nmax,title]]];
	f /. HarmonicRules
];

MellinInverseSinglet[f_] := Module[{temp,fx,f1,f1x,f2,f2x,f3,f3x,f4,f4x},
	(* Mellin inversion *)
	(* f1 = S[1,n]/n^2;
	f1x = -PolyLog[2,x]+Zeta[2];
	f2 = S[1,n]^2/n^2;
	f2x = -(1/3) \[Pi]^2 Log[1-x]+Log[1-x]^2 Log[x]+(2 Log[1-x]-Log[x]) PolyLog[2,x]+2 PolyLog[3,1-x]+2 PolyLog[3,x]-2 Zeta[3];
	f3 = S[1,n]^3/n^2;
	f3x = -(\[Pi]^4/15)-1/2 \[Pi]^2 Log[1-x] Log[x]+Log[1-x]^3 Log[x]+3/2 Log[1-x]^2 Log[x]^2+3 Log[1-x]^2 PolyLog[2,1-x]+1/2 (\[Pi]^2+6 Log[1-x] Log[x]-Log[x]^2) PolyLog[2,x]-6 Log[1-x] PolyLog[3,1-x]+3 Log[x] PolyLog[3,1-x]+2 Log[x] PolyLog[3,x]+6 PolyLog[4,1-x]-3 PolyLog[4,x]+6 PolyLog[2,2,x]-3 Log[x] Zeta[3];
	f4 = S[1,n]^2/n;
	f4x = 2 H[1,1,x]-Log[1-x] Log[x]-PolyLog[2,x];

	temp = (f - Coefficient[f,f3] f3 ) //Expand;
	temp = (temp - Coefficient[f,f2] f2 ) //Expand;
	temp = (temp - Coefficient[f,f1] f1 ) //Expand;
	temp = (temp - Coefficient[f,f4] f4 ) //Expand;
	fx = InvMellin[ temp //. QCDConstantsRules /. ZetaRules /. n -> n+1, n, x] /. InvMellinRules /. hrep;
	fx + Coefficient[f,f1] f1x + Coefficient[f,f2] f2x +  Coefficient[f,f3] f3x + Coefficient[f,f4] f4x //ReduceConstants*)
	ReduceToHBasis[InvMellin[ f //. QCDConstantsRules /. ZetaRules /. n -> n+1, n, x] /. InvMellinRules ] /.hrep
];


RandomBasis[fsmallN_, flargeN_, subleadingfs_]:=Module[{basis, r1, r2},
	basis = {};
	r1 = RandomChoice[subleadingfs];
	r2 = RandomChoice[DeleteCases[subleadingfs,Alternatives@@{r1}]];
	Do[
		basis = Append[basis, Flatten[{fsmallN[[nf]], flargeN[[nf]], r1[[nf]], r2[[nf]]}]],
	{nf,1,Length[fsmallN]}];
	basis
];

RandomFit[fsmallN_, flargeN_, subleadingfs_ , moments_, N0asy_, N1asy_, Ninf_]:=Module[{fitbasis, Nfmax, Nfmin},
	If[Length[fsmallN]==4, Nfmin=0, Nfmin=1];
	Nfmax=3;
	fitbasis = RandomBasis[fsmallN, flargeN, subleadingfs,Nfmin, Nfmax];
	AdFitSinglet[moments,Nfmin, Nfmax, N0asy,N1asy, Ninf, fitbasis]
];


DoRandomFits[fsmallN_, flargeN_, subleadingfs_ , moments_, N0asy_, N1asy_, Ninf_, Nfits_]:= Module[{FittedRandom},
	FittedRandom = {};
	Do[
		FittedRandom = Append[FittedRandom, RandomFit[fsmallN, flargeN, subleadingfs, moments, N0asy,N1asy, Ninf]];
	,{i,0,Nfits}];
	FittedRandom
];

DoAllFits[fsmallN_, flargeN_, subleadingfs_ , moments_, N0asy_, N1asy_, Ninf_, isNNLO_: False]:= Module[{Fittedlist, product, fitbasis, Nfmax,Nfmin},
	Fittedlist = {};
	product = DeleteCases[MapThread[ DeleteDuplicates[Sort[#]] &,{ Tuples[subleadingfs,2]}]  // DeleteDuplicates , {_}];
	Do[
		Nfmax=3;
		fitbasis = RandomBasis[fsmallN, flargeN, product[[i]]];
		If[Length[fsmallN]==4,
			Nfmin=0,
			If[isNNLO,
				Nfmax=2; Nfmin=0;,
				Nfmin=1; fitbasis = Join[{{0,0,0,0}}, fitbasis];
			];
		];
		Fittedlist = Append[Fittedlist, AdFitSinglet[moments,Nfmin, Nfmax, N0asy,N1asy, Ninf, fitbasis]];
	,{i,1,Length[product]}];
	Fittedlist
];

InterploatedFit[fsmallN_, flargeN_, subleadingfs_ , moments_, N0asy_, N1asy_, Ninf_, newN_]:=Module[{newmoments, momentslist, nvals, Nmin,fasy},
	fasy =  N0asy + N1asy + Ninf;
	Nmin=2;
	nvals = Range[Nmin, 2 Length[moments],2];
	momentslist = MapThread[#1 - fasy //.n -> #2 &, {moments, nvals}];
	newmoments = BuildNewMoment[momentslist, nvals, #] + fasy /. n -> # & /@ newN;
	momentslist = Append[moments, newmoments] // Flatten;
	momentslist = MapThread[{#1, #2}&, {Append[nvals, newN] // Flatten, momentslist}];
	DoAllFits[fsmallN, flargeN, subleadingfs, momentslist, N0asy, N1asy, Ninf]
];


Needs["FunctionApproximations`"]

(*
OldMomentListInterpolator[moments_,nvals_, NF_]:=Module[{momentslist, basis},
		momentslist[n_] := Indexed[Coefficient[moments,nf,NF] //. QCDConstantsRules /. ZetaRules /. InverseHarmonicRules, IntegerPart[n/2]];
		RationalInterpolation[momentslist[n], {n, 2, 1}, nvals]
];

MomentListInterpolator[moments_,nvals_, NF_]:=Module[{momentslist, basis},
		momentslist = Coefficient[moments,nf,NF] //. QCDConstantsRules /. ZetaRules /. InverseHarmonicRules;
		basis ={PolyGamma[0,n-1],PolyGamma[1,n-1], 1/(n-1), n/(n-1)};
		Fit[MapThread[{#1,#2 /. n -> #1} &, {nvals, momentslist}], basis, n]
];
*)
(*
MomentListInterpolator[moments_,x_, NF_]:=Module[{momentslist, basis1,basis2, coordinates},
	momentslist = Coefficient[moments,nf,NF] //. QCDConstantsRules /. ZetaRules /. InverseHarmonicRules;
	coordinates = MapThread[{#1,#2 /. n -> #1 }&,{x, momentslist}];
	h[i_] := Transpose[coordinates][[2]][[i]];
	(*h[i_, 1] := (h[i+1]-h[i])/2 (n - x[[i]]) + h[i];*)
	basis1 = {1/((n-1)n),Log[n-1]};
	basis2 = {1/((n-1)n),Log[n]};
	h[1, 1] = Fit[{coordinates[[1]], coordinates[[2]]}, basis1, n];
	h[2, 1] = Fit[{coordinates[[2]], coordinates[[3]]}, basis1, n];
	h[3, 1] = Fit[{coordinates[[3]], coordinates[[4]]}, basis2, n];
	h[j_, 2] := h[j+1] + (x[[j+2]] - x[[j]])/((n-x[[j]])/(h[j+1,1]- h[j+1]) + (x[[j+2]]-n)/(h[j,1]- h[j+1]));
	h[j_, 3] := h[j+1,1] + (x[[j+3]] - x[[j]])/((n-x[[j]])/(h[j+1,2]- h[j+1,1]) + (x[[j+3]]-n)/(h[j,2]- h[j+1,1]));
	h[1,3]
];
*)

slice[i_, d_, nvals_]:=Module[{xi},
	xi = {};
	Do[xi=Append[xi, nvals[[j]]],{j,i+1,i+d}];
	xi
]

MomentListInterpolator[moments_,nvals_, NF_]:=Module[{d, momentslist, coordinates, ci, N, p, xi, lambda, num, den, basis},
	(*
	Rational interpolation whitout poles, from https://d-nb.info/1038484707/34
	d = the number of points used in the first step
	*)
	d = 3;
	momentslist = Coefficient[moments,nf,NF] //. QCDConstantsRules /. ZetaRules /. InverseHarmonicRules;
	coordinates = MapThread[{#1,#2 /. n -> #1 }&,{nvals, momentslist}];
	N = Length[coordinates];
	num = 0;
	den = 0;
	basis = {PolyGamma[0,n-1]/n,PolyGamma[1,n-1]/n, 1/(n-1)};
	Do[
		xi = slice[i,d,nvals];
		ci = slice[i,d,coordinates];
		lambda = (-1)^i / Times @@ ((n-#)& /@ xi);
		(* p = InterpolatingPolynomial[ci,n]; *)
		(* p = RationalInterpolation[f[n], {n, Length[xi]-2, 1}, xi]; *)
		p = Fit[ci, basis, n];
		num += lambda p;
		den += lambda;
	,{i,0,N-d}];
	num / den
];

BuildNewMoment[moments_, nvals_, newN_]:=Module[{momentslist, momentsInterpolator},
	momentsInterpolator = 0;
	Do[
		momentsInterpolator += nf^ NF MomentListInterpolator[moments,nvals, NF]
	,{NF,0,3}];
	momentsInterpolator /. n -> newN
];

InterploatedFit[fsmallN_, flargeN_, subleadingfs_ , moments_, N0asy_, N1asy_, Ninf_, newN_]:=Module[{newmoments, momentslist, nvals, Nmin,fasy},
	fasy =  N0asy + N1asy + Ninf;
	Nmin=2;
	nvals = Range[Nmin, 2 Length[moments],2];
	momentslist = MapThread[#1 - fasy //.n -> #2 &, {moments, nvals}];
	newmoments = BuildNewMoment[momentslist, nvals, #] + fasy /. n -> # & /@ newN;
	momentslist = Append[moments, newmoments] // Flatten;
	momentslist = MapThread[{#1, #2}&, {Append[nvals, newN] // Flatten, momentslist}];
	DoAllFits[fsmallN, flargeN, subleadingfs, momentslist, N0asy, N1asy, Ninf]
];



ListMellinInverse[f_] := ReduceToHBasis[MapThread[InvMellin[# /. n -> n+1, n,x] &, {f}] /. InvMellinRules]  /. hrep /. Delta[x__] -> 0 /. ZetaRules;


ComputeArc[f_, NF_,  thrlow_, thrhigh_] := Module[{},
  ArcLength[pifact  f /. nf -> NF , {x, thrlow, 1 - thrhigh}]
];

ComputeArcList[f_, NF_,  thrlow_, thrhigh_] := Module[{},
  Parallelize[MapThread[ComputeArc[#,  NF,thrlow, thrhigh] &, {f}]]
];

SelectSmoothCandidates[flist_, NF_, thrlow_, thrhigh_]:=Module[{goodcadidates, arcdist, arcavg, arcstd},
	arcdist = ComputeArcList[flist,NF,thrlow,thrhigh];
	arcavg = Mean[arcdist];
	arcstd = StandardDeviation[arcdist];
	goodcadidates = Select[flist, ComputeArc[# , NF , thrlow, thrhigh] - arcavg  <= arcstd &];
	Print["Smooth candidates: ", goodcadidates //Length, " out of ", flist //Length];
	goodcadidates
];


SelectCherries[candidates_, cherryfunctions_]:=Module[{cherry, cherrylist},
	cherrylist = MapThread[Cases[candidates, __ + __ #,1] &, {cherryfunctions}];
	cherry = cherrylist // DeleteDuplicates //Flatten // DeleteDuplicates;
	Print["Selected cherries are: ", cherry // Length, " out of ", candidates // Length];
	ListMellinInverse[cherry /. S2m2 -> S[2,n-2]/n]
];

SelectCherriesSpecial[candidates_, cherryfunctions_]:= Module[{cherry, cherrylist},
	cherrylist = MapThread[candidates[[ Flatten[Position[candidates /.  S[1,n-2]/n -> S1m2 /. nf^3 -> 0 /. nf^2 -> 0 // Expand, __ + __ #]] ]]&, {cherryfunctions}];
	cherry = cherrylist // DeleteDuplicates //Flatten // DeleteDuplicates;
	Print["Selected cherries are: ", cherry // Length, " out of ", candidates // Length];
	ListMellinInverse[cherry /. S1m2 -> S[1,n-2]/n]
];
