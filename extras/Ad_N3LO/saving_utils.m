(* ::Package:: *)

LogFunctionRules = {
   	Lm11[x_] -> lm11[x, S1],
   	Lm12[x_] -> lm12[x, S1, S2],
   	Lm13[x_] -> lm13[x, S1, S2, S3],
   	Lm14[x_] -> lm14[x, S1, S2, S3, S4],
   	Lm15[x_] -> lm15[x, S1, S2, S3, S4, S5],
   	Lm11m1[x_] -> lm11m1[x, S1],
   	Lm12m1[x_] -> lm12m1[x, S1, S2],
   	Lm13m1[x_] -> lm13m1[x, S1, S2, S3],
   	Lm14m1[x_] -> lm14m1[x, S1, S2, S3, S4],
   	S13m1[x_] ->  S1^3/n,
   	S12m1[x_] ->  S1^2/n,
	   S13m2[x_] ->  S1^3/n^2,
   	S12m2[x_] ->  S1^2/n^2,
   	S11m2[x_] ->  S1/n^2,
   	SuppLog[x_] -> (1/(n-1)-1/(n))
};
InverselargeNrules ={S3m2 -> (-(((-1+2 n) (1-n+n^2))/((-1+n)^3 n^3))+1/2 (2 S[3,n]-2 Zeta[3])+Zeta[3])/n , S2m2 -> ((-1+2 n-2 n^2)/((-1+n)^2 n^2)+S[2,n])/n , S1m2 ->((1-2 n)/((-1+n) n)+S[1,n])/n};
largeNrules ={ (-(((-1+2 n) (1-n+n^2))/((-1+n)^3 n^3))+1/2 (2 S[3,n]-2 Zeta[3])+Zeta[3])/n-> S3m2,  ((-1+2 n-2 n^2)/((-1+n)^2 n^2)+S[2,n])/n -> S2m2, ((1-2 n)/((-1+n) n)+S[1,n])/n -> S1m2};

exportMe[common_, fitA_, fitB_, fp_, NF_] := Module[{me, meA, meB, out},
			 me = Coefficient[common, nf, NF] //. QCDConstantsRules  /. LogFunctionRules /. harmonics // N;
             meA = Coefficient[fitA, nf, NF] //. QCDConstantsRules  /. LogFunctionRules /. harmonics // N;
             meB = Coefficient[fitB, nf, NF] //. QCDConstantsRules  /. LogFunctionRules /. harmonics // N;
             Print["writing to ", fp];
             out = OpenWrite[fp];
             WriteString[out, ToString@CForm@me];
             WriteString[out, "\n"];
             WriteString[out, ToString@CForm@meA];
             WriteString[out, "\n"];
             WriteString[out, ToString@CForm@meB];
             Close[out];
];

exportMeList[common_, fitlist_, fp_, NF_] := Module[{commoncoeff, coefflist, avg, out},
   			 commoncoeff = Coefficient[common, nf, NF] //. QCDConstantsRules /. LogFunctionRules /. harmonics // N;
   			 coefflist = Coefficient[fitlist, nf, NF] //. QCDConstantsRules /. LogFunctionRules /. harmonics /. largeNrules // N;
   			 avg = Mean[coefflist] // Simplify;
   			 Print["writing to ", fp];
                out = OpenWrite[fp];
                WriteString[out, ToString @ CForm @ commoncoeff];
                WriteString[out, "\n"];
                WriteString[out, ToString @ CForm @ avg];
                WriteString[out, "\n"];
                Do[
                  WriteString[out, ToString@ CForm @ coefflist[[i]]];
                  WriteString[out, "\n"];
                ,{i, 1, Length[coefflist]}];
                Close[out];
];
here = NotebookDirectory[];
storepath = StringJoin[here, "parametrised_ad/"];
