For further details see also:
file:///Volumes/Git_Workspace/physicstools/NN3PDF/eko/doc/build/html/theory/N3LO_ad.html

## Singlet sector

We  construct the approximation for the singlet quantities : gqq3ps, ggg3, ggq3, gqg3.
	*	Low N even (2, 4, 6, 8) moments are taken from  2111.15661.
   *   	Large N limit is given for the diagonal entries by the QCDcusp (see after Eq. 4 of 2111.15661).
		   The approximation has accuracy O(1/N) and fix completely the terms  Log[N].
	     	The calculation of the QCD cusp is the same as the singlet sector and is provided in 1911.10174, see ancillary file and "qcd_cusp.m"
   	   	But looking at the expression of gqq0 we believe A_k * S[1, N],  to be more accurate expansion.
        The coeffiencient of the delta function is obtained from 2205.04493, see "gamma_thr.m"
   		The off-diagonal term contain a contribution coming from 0912.0369 fixing Log[1-x]^k k=5,4
   * 	The small-x limit contains two contributions:
		-	the BKFL limit [1/x Log[x] ]  which fixes the leading contributions for N=1,
			see 1805.06460 (section 2.3, 2.4). They fix:
				 * gg: 1/(N-1)^4 +  1/(N-1)^3,
				 * gq: 1/(N-1)^4,
				 * qg:  1/(N-1)^3,
				 * qq_ps: 1/(N-1)^3
				NOTE : in this paper a different  convention is used for the Mellin transform. (x^N)
		- 	the single logs  [Log[x] ... Log[x]^6 ]  which fixes the leading contributions for N=0  O(1/N^4).
			The following coefficients are provided: 1/N^7, 1/N^6, 1/N^5
			These terms are sub-leading w.r.t. to the BFKL one in the small-x region.
* 	The part proportional to nf^3, is known exact, see 1610.07477 and its ancillary file
* 	The remaining parts are unknown and thus fitted
* 	Note gqq3ps and gqg3 do not contain the nf^0 part.


## Non singlet sector
We  construct the approximation for the non singlet quantities : gqq3nsp, gqq3nsm, gqq3nss.
NOTE: 	s is defined as ns_s = ns_v - ns_m. It contains parts proportional only to nf and nf^2.
		It vanishes in the large nc limit. (N -> Infinity, it goes to 0)

	*	Limits for N = 0 are taken from 2202.10362, see ancillary file provided and Eq. 3.3, 3.8, 3.9, 3.10 :
  		All the expression have the accuracy O (1/N) and are given in the large colour limit, where p,m,v are all equal .
	* 	Low N odd (1, 3, 5, 7,9,11,13,15,(17)) moments are taken from  1707.08315.
	*   Large N limit is given in Eq 2.17 of 1707.08315.
		The approximation has accuracy O(1/N^2) and fix the terms  Log[N], Log[N]/N, 1/N.
		According to 1707.08315 terms with Log[N]^k , k> 2 are not present in the expansion.
	     	- The calculation of the QCD cusp is the same as the singlet sector and is provided in 1911.10174, see ancillary file and "qcd_cusp.m".
			- The coeffiencient of the delta function is obtained from 2205.04493, see "gamma_thr.m" and it's more accurate than
			  B4 of  1707.08315 ( which is only in the large N limit). In the fit we include a constant term to improve the fit convergernce
			  and we check that the final coefficient of the delta function do not differ much from the given input.
   	     	- Looking at the expression of gqq0 we believe A_k * S[1, N] , S[1,N]/N 1/N to be more accurate expansion.
	* 	The part proportional to nf^3, common for all the p,m,v (s is then vanishing) and is taken exact for Eq 3.6 of 1610.07477 and its ancillary file
	* 	For m,p: the part proportional to nf^2 are also given exact form Eq 2.12, 3.1, 3.2, 3.3, 3.4 of 1610.07477 and its ancillary file
	* 	For s: The part proportional to nf^2 are also taken exact form Eq 3.5 of 1610.07477 and its ancillary file
