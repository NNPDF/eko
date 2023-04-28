(* ::Package:: *)

PlotFittedTable[DataTable_, f_, min_,max_, title_]:=Module[{g1, g2},
	(*plot fit and exact series *)
	g1 = ListPlot[DataTable, PlotMarkers->{Automatic, 10}];
	g2 = Plot[f, {n,min,max}, PlotStyle->Red];
	Show[g1,g2, AxesOrigin->{0,0}, PlotRange->{{min,max}, {Min[DataTable], Max[DataTable]}}, PlotLabel->title]
];


PlotXspaceSinglet[title_, fitted_, asyN0_, asyN1_, asyNinf_,  NF_: 4, xmin_: 0.00001, xhigh_: 0.00001] := Module[{Bfkl, DoubleLog,Fitted,Asyinf,Total,legend, flist, g1,g2,g3, pifact, flistinv},
	(* x space plot *)
	DoubleLog = MellinInverseSinglet[asyN0];
	Bfkl = MellinInverseSinglet[asyN1];
	Fitted = MellinInverseSinglet[fitted] /. Delta[__] -> 0;
	Asyinf = MellinInverseSinglet[asyNinf] /. Delta[__] -> 0;
	legend = {"small-x (2202.10362)", "BFKL", "Fitted moments", "Large N limit", "Total"};
	Total = DoubleLog + Bfkl + Fitted + Asyinf;
	pifact = 1/(4 Pi)^4;
	flist = { - DoubleLog pifact, - Bfkl pifact, - Fitted pifact, - Asyinf pifact, - Total pifact} /. nf -> NF;
	Print[ "Limit for x=1 : ", ReduceConstants[Normal[Series[ - Total pifact /. nf -> NF, {x,1,1}, Assumptions->{x<1}]], ToKnownConstants-> True]];
	g1 = LogLinearPlot[
		flist, {x,xmin, 1},
		PlotLegends->legend, PlotLabel->title, ImageSize->Medium, AxesLabel->{"x",""}
	];
	g2 = Plot[
		flist , {x,xmin, 1},
		PlotLegends->legend, PlotLabel->title, ImageSize->Medium, AxesLabel->{"x",""}
	];
	flistinv = { - DoubleLog pifact, - Bfkl pifact, - Fitted pifact, - Asyinf pifact, - Total pifact} /. nf -> NF /. x -> 1-x;
	g3 = LogLinearPlot[
		flistinv, {x,xhigh, 1},
		PlotLegends->legend, PlotLabel->title, ImageSize->Medium, AxesLabel->{"1-x",""}
	];
	GraphicsRow[{g1,g2,g3}, ImageSize->Full]
];


PlotRelativeDiffSinglet[fexa_, ffitted_, NF_, Nmin_, Nmax_]:=Module[{difflist,nmax, ffitednum, fexanum},
	(* N space relative difference plot *)
	ffitednum = ffitted //. QCDConstantsRules /. ZetaRules;
	fexanum = fexa //. QCDConstantsRules /. ZetaRules;
	difflist = Table[( Coefficient[ffitednum,nf,NF] - fexanum)/fexanum *100, {n,Nmin,Nmax}];
	ListPlot[MapThread[{#1, #2}&, {Range[Nmin,Nmax], difflist}], PlotLabel-> StringForm["% difference (fitted-exact)/exact, NF^``",NF], ImageSize->Medium]
];

PlotRatioToBest[v1_, v2_, t1_, t2_, NF_, title_, Nmin_: 1.01, Nmax_: 10, Nstep_: 0.01] :=  Module[{ref,Nlist,l0, l1,l2,l3,l4},
	ref = (v1 + v2)/2;
	Nlist =Range[Nmin, Nmax, Nstep];
	l1 = MapThread[{#, v1 / ref /. nf -> NF /. n -> # // N} &, {Nlist}];
	l2 = MapThread[{#, v2 / ref /. nf -> NF /. n -> # // N} &, {Nlist}];
	l3 = MapThread[{#, t1 / ref /. nf -> NF /. n -> # // N} &, {Nlist}];
	l4 = MapThread[{#, t2 / ref /. nf -> NF /. n -> # // N} &, {Nlist}];
	l0 = MapThread[{#,  1 /. n -> # // N} &, {Nlist}];
	ListPlot[
		{l1,l2,l3,l4, l0},
		AxesLabel->{"N", "Ratio to best"},
		PlotLegends->{"Small-x", "Large-x", "Test1", "Test2"},
		PlotLabel->title,
		PlotStyle->{ColorData[97,"ColorList"][[1]], ColorData[97,"ColorList"][[2]], ColorData[97,"ColorList"][[3]], ColorData[97,"ColorList"][[4]],Gray},
		ImageSize->Large
	]
];


PlotInterpolation[moments_, newN_, NF_]:=Module[{nvals, momentsInterpolator, gintmom, gmom, gint},
	nvals = {2, 4, 6, 8};
	momentsInterpolator = MomentListInterpolator[moments,nvals, NF];
	gintmom = ListPlot[
		MapThread[{#, Coefficient[ BuildNewMoment[moments, nvals, #], nf, NF]} &,
			{newN}], PlotStyle -> Blue, PlotMarkers -> {"*", Medium}];
	gmom = ListPlot[
		MapThread[{#1, Coefficient[#2/. n -> #1 //. QCDConstantsRules /. ZetaRules, nf, NF] } &,
		{nvals, moments}], PlotStyle -> Red, PlotMarkers -> {Automatic, Small}];
	gint = Plot[momentsInterpolator, {n,1.5,9}];
	Show[gint, gintmom, gmom, PlotLabel->StringForm["Prop to nf^``.",NF], ImageSize->Full]
];


clist = ColorData[97, "ColorList"];
myStyle[nc_] := {{clist[[nc]], Dotted}, {clist[[nc]], Dashed}, {clist[[nc]]}};
defaultlegends = {"large-N","small-N"};

pifact = -x /(4 Pi)^4 0.2^4;

PlotList[fitted_, fasy_, NF_, color_]:=Module[{},
	Parallelize[MapThread[
		Plot[ pifact (# + fasy) /. nf -> NF , {x, 0, 1}, PlotStyle->clist[[color]]] &,
	 {fitted}]]
];

PlotListLegend[flist_, fasy_, NF_, color_, legend_] := Module[{plot, legendplot},
   plot = PlotList[flist, fasy, NF, color];
   legendplot = Plot[Null, {x, 0, 1}, PlotLegends -> {legend}, PlotStyle -> clist[[color]]];
   Show[Join[plot, {legendplot} ] // Flatten, ImageSize -> Large]
];

LogPlotList[fitted_, fasy_, NF_, color_]:=Module[{},
	Parallelize[
		MapThread[
			LogLinearPlot[ pifact (# + fasy) /. nf -> NF , {x, 10^-7, 1}, PlotStyle->clist[[color]]] &,
		{fitted}
	]]
];

ShowPlots[fit1_, fit2_, fasy_, NF_, legends_]:=Module[{g1list, g2list, g1listlog, g2listlog, gtotlog, gtot},
	g1list = PlotList[fit1, fasy, NF, 1];
	g1listlog = LogPlotList[fit1, fasy, NF, 1];
	g2list = PlotList[fit2, fasy, NF, 2];
	g2listlog = LogPlotList[fit2, fasy, NF, 2];
	gtot = ShowLegends[Join[g2list, g1list] // Flatten, Plot, legends, {1,2}];
	gtotlog = ShowLegends[Join[g2listlog, g1listlog] // Flatten, LogLinearPlot, legends, {1,2}];
	{gtotlog, gtot}
];

ShowLegends[plotlist_, plot_, legends_, colors_]:=Module[{fakelegend},
	fakelegend = MapThread[ plot[Null, {x,0,1}, PlotLegends->{#1}, PlotStyle->clist[[#2]]]&, {legends, colors}];
	Show[Join[plotlist, fakelegend] // Flatten, ImageSize -> Large]
]


LinearPlotErrorBand[fasy_,  avg_, sigma_, NF_, color_ ]:=Module[{ftot},
	ftot = fasy + avg ;
	Plot[{ pifact (ftot - sigma) /. nf -> NF, pifact (ftot + sigma)  /. nf -> NF , pifact (ftot)  /. nf -> NF}, {x, 0, 1},
		Filling -> {1 -> {2}}, PlotStyle -> myStyle[color]]
];

LogPlotErrorBand[fasy_,  avg_, sigma_, NF_, color_ ]:=Module[{ftot},
	ftot = fasy + avg ;
	LogLinearPlot[{ pifact (ftot - sigma) /. nf -> NF, pifact (ftot + sigma)  /. nf -> NF , pifact (ftot)  /. nf -> NF}, {x, 10^-7, 1},
		Filling -> {1 -> {2}}, PlotStyle -> myStyle[color]]
];

PlotErrorBand[fasy_, flist_, NF_, color_ ]:=Module[{mean, sigma,g1, g1log},
	{mean, sigma} = Parallelize[{
		(Mean[flist] // Simplify)  /. hrep /. InvMellinRules,
		(StandardDeviation[flist] // Simplify ) /. hrep /. InvMellinRules
	}];
	Parallelize[{
		LinearPlotErrorBand[fasy, mean, sigma, NF, color],
		LogPlotErrorBand[fasy, mean, sigma, NF, color]
	}]
];


PlotRelativeDiff[ref_, fitted_, NF_, Nmin_, Nmax_]:=Module[{rules, f1, fanalytic},
	rules = Flatten[Join[QCDConstantsRules, ZetaRules, InverseHarmonicRules]];
	f1 = Coefficient[fitted //. rules, nf,NF];
	fanalytic = ReduceToBasis[ref] //. rules ;
	Plot[(f1-fanalytic)/ fanalytic * 100, {n, Nmin,Nmax}, ImageSize->Large, PlotRange->All, PlotLabel-> StringForm["% difference (fitted-exact)/exact, NF^``",NF]]
];

PlotRelativeDiffIterate[ref_, flist_, NF_, Nmin_, Nmax_]:=Module[{glist, gmean, avg},
	glist = MapThread[PlotRelativeDiff[ref, #, NF, Nmin, Nmax]&, {flist}];
	avg = Mean[flist];
	gmean = PlotRelativeDiff[ref, avg, NF, Nmin, Nmax] /. _?ColorQ -> Red;
	Show[Join[glist, {gmean}] // Flatten ]
];


PlotListN[common_, fitlist_, NF_, fmin_, fmax_]:=Module[{commoncoeff, coefflist, avg, plot, plotlist},
    commoncoeff = Coefficient[common, nf, NF] //. QCDConstantsRules;
    coefflist = Coefficient[fitlist, nf, NF] //. QCDConstantsRules;
    plotlist = MapThread[Plot[# + commoncoeff /. InverselargeNrules/. InverseHarmonicRules, {n,1,6}, PlotRange->{All,{fmin,fmax}}]&, {coefflist}];
    avg = Mean[coefflist];
    plot = Plot[
       {avg + commoncoeff/. InverselargeNrules /. InverseHarmonicRules, commoncoeff /. InverseHarmonicRules},{n,1,6},
       PlotRange-> {All,{fmin,fmax}},
       PlotStyle->{Red,Orange}
    ];
    Join[{{plot}, plotlist}]// Flatten
];
