(* ::Package:: *)

(* From https://arxiv.org/src/1911.10174v1/anc/HKM_GammaC.txt *)
(*
Notation:
*)
GluonConstants ={
	CR -> ca,
	C4A ->d4AA/na,
	C4F-> d4RA/na
};
QuarkConstants ={
	CR -> cf,
	C4A ->d4RA/nr,
	C4F-> d4RR/nr
};
SytnaxReplacemnts = {CA-> ca, TF -> tr, NF -> nf, CF-> cf};
QCDcusp = aS*CR+(67*aS^2*CA*CR)/36+(245*aS^3*CA^2*CR)/96+(42139*aS^4*CA^3*CR)/10368-(5*aS^2*CR*NF*TF)/9-(209*aS^3*CA*CR*NF*TF)/216-(24137*aS^4*CA^2*CR*NF*TF)/10368-(55*aS^3*CF*CR*NF*TF)/48-(17033*aS^4*CA*CF*CR*NF*TF)/5184+(143*aS^4*CF^2*CR*NF*TF)/288-(aS^3*CR*NF^2*TF^2)/27+(923*aS^4*CA*CR*NF^2*TF^2)/5184+(299*aS^4*CF*CR*NF^2*TF^2)/648-(aS^4*CR*NF^3*TF^3)/81-(aS^4*C4A*Zeta[2])/2-(aS^2*CA*CR*Zeta[2])/2-(67*aS^3*CA^2*CR*Zeta[2])/36-(5525*aS^4*CA^3*CR*Zeta[2])/1296+aS^4*C4F*NF*Zeta[2]+(5*aS^3*CA*CR*NF*TF*Zeta[2])/9+(635*aS^4*CA^2*CR*NF*TF*Zeta[2])/324+(55*aS^4*CA*CF*CR*NF*TF*Zeta[2])/48-(19*aS^4*CA*CR*NF^2*TF^2*Zeta[2])/162+(aS^4*C4A*Zeta[3])/6+(11*aS^3*CA^2*CR*Zeta[3])/24+(1309*aS^4*CA^3*CR*Zeta[3])/432-(aS^4*C4F*NF*Zeta[3])/3-(7*aS^3*CA*CR*NF*TF*Zeta[3])/6-(361*aS^4*CA^2*CR*NF*TF*Zeta[3])/54+aS^3*CF*CR*NF*TF*Zeta[3]+(29*aS^4*CA*CF*CR*NF*TF*Zeta[3])/9+(37*aS^4*CF^2*CR*NF*TF*Zeta[3])/24+(35*aS^4*CA*CR*NF^2*TF^2*Zeta[3])/27-(10*aS^4*CF*CR*NF^2*TF^2*Zeta[3])/9+(2*aS^4*CR*NF^3*TF^3*Zeta[3])/27-(11*aS^4*CA^3*CR*Zeta[2]*Zeta[3])/24+(7*aS^4*CA^2*CR*NF*TF*Zeta[2]*Zeta[3])/6-aS^4*CA*CF*CR*NF*TF*Zeta[2]*Zeta[3]-(3*aS^4*C4A*Zeta[3]^2)/2-(aS^4*CA^3*CR*Zeta[3]^2)/16+(11*aS^3*CA^2*CR*Zeta[4])/8+(451*aS^4*CA^3*CR*Zeta[4])/64-(11*aS^4*CA^2*CR*NF*TF*Zeta[4])/24-(11*aS^4*CA*CF*CR*NF*TF*Zeta[4])/8-(7*aS^4*CA*CR*NF^2*TF^2*Zeta[4])/12+(aS^4*CF*CR*NF^2*TF^2*Zeta[4])/2+(55*aS^4*C4A*Zeta[5])/12-(451*aS^4*CA^3*CR*Zeta[5])/288-(5*aS^4*C4F*NF*Zeta[5])/3+(131*aS^4*CA^2*CR*NF*TF*Zeta[5])/72+(5*aS^4*CA*CF*CR*NF*TF*Zeta[5])/4-(5*aS^4*CF^2*CR*NF*TF*Zeta[5])/2-(31*aS^4*C4A*Zeta[6])/8-(313*aS^4*CA^3*CR*Zeta[6])/96;
QCDcusp4 =Coefficient[QCDcusp,aS,4] /. SytnaxReplacemnts //N;
QCDcusp3 =Coefficient[QCDcusp,aS,3] /. SytnaxReplacemnts //N;
QCDcusp2 =Coefficient[QCDcusp,aS,2] /. SytnaxReplacemnts //N;
QCDcusp1 =Coefficient[QCDcusp,aS,1] /. SytnaxReplacemnts //N;
