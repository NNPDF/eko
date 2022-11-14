set terminal pdfcairo dashed enhanced


set logscale x
set xrange [*:*] reverse
set key top left Left
#set xzeroaxis



c=4./3./3.

zeta2=1.6449340668482264365
zeta3=1.2020569031595942854
# NNLO
cqg22(nf)=-3.**2*nf/3./pi**3*14./9.
cgg22(nf)=-(3**3*(-395+99*zeta2+54*zeta3)+nf*3*(3-8./3.)*(18*zeta2-71))/108./pi**3 - 4./9.*cqg22(nf)
# N3LO
cqg33(nf)=3.**3*nf/6./pi**4*(82./81.+2*zeta3)
cgg34=-(3./pi)**4 *zeta3/3.
#before corrections
#cgg33(nf)=(3**4*(-1205./162.+67*zeta2/36.+zeta2**2/4.-33*zeta3/2.)+nf*3**3*(-233./162.+13*zeta2/36.+5*zeta3/3.)+nf*12*(617./243.-13*zeta2/18.+2*zeta3/3.))/2./pi**4
#cgg33MS(nf)=(3**4*(-1205./162.+67*zeta2/36.+zeta2**2/4.-55*zeta3/6.)+nf*3**3*(-233./162.+13*zeta2/36.+zeta3/3.)+nf*12*(617./243.-13*zeta2/18.+2*zeta3/3.))/2./pi**4
#after corrections
cgg33Q0(nf)=(3**4*(-1205./162.+67*zeta2/36.+zeta2**2/4.-77*zeta3/6.)+nf*3**3*(-233./162.+13*zeta2/36.+zeta3   )+nf*12*(617./243.-13*zeta2/18.+2*zeta3/3.))/2./pi**4
cgg33MS(nf)=(3**4*(-1205./162.+67*zeta2/36.+zeta2**2/4.-11*zeta3/2.)+nf*3**3*(-233./162.+13*zeta2/36.-zeta3/3.)+nf*12*(617./243.-13*zeta2/18.+2*zeta3/3.))/2./pi**4



gr='rgbcolor "#aaaaaa"'
bl='rgbcolor "#888888"'
p ='rgbcolor "#cc33cc"'
g ='rgbcolor "#33cc33"'
dg='rgbcolor "#119911"'
b ='rgbcolor "#4499ff"'
y ='rgbcolor "#ffdd11"'
dy='rgbcolor "#ffbb11"'
r ='rgbcolor "#ff9944"'
re='rgbcolor "#ff4433"'
w ='rgbcolor "#ffffff"'

ABFf  = '../hell/output/abf/abf.dat'
ABFf1 = '../hell/output/abf/abf-pgg-ll.dat'
ABFf2 = '../hell/output/abf/abf-pgq-ll.dat'

set encoding utf8
set terminal pdfcairo dashed font "Palatino, 12" enhanced lw 0.8 dl 1 size 3.1,2.3
set xrange [1:1e-9]
set logscale x
set xlabel "x"
set key center top Left reverse
set xtics ("1"1,"10^{-1}"0.1,"10^{-2}"0.01,"10^{-3}"0.001,"10^{-4}"0.0001,"10^{-5}"0.00001,"10^{-6}"0.000001,"10^{-7}"0.0000001,"10^{-8}"0.00000001,"10^{-9}"0.000000001)
#
set style lines 1 lc @bl lw 2 dt 3
set style lines 2 lc @bl lw 2 dt 2
set style lines 3 lc @bl lw 2 dt 5
set style lines 4 lc @g  lw 2 dt 3
#set style lines 8 lc @dg lw 1 dt 3
set style lines 5 lc @p  lw 2 dt 2
set style lines 6 lc @b  lw 2 dt 5
set style lines 77 lc @y  lw 2 dt 2
set style lines  7 lc @dy lw 2 dt 2
set style fill transparent solid 0.25 noborder

set macro
eg = '$30'
eq = '$31'
egLL = '$37'



set output 'plots/plot_P_nf4_paper.pdf'
set key at graph 0.2,0.95
set ylabel 'x P_{gg}(x)'
as=20
set yrange [0:0.02*as]
#set yrange [0:0.1]
set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
file  = 'output/P_as0'.as.'_nf4.dat'
#
plot file  using 2:6        ls 1 with l title 'LO', \
     ''    using 2:10       ls 2 with l title 'NLO', \
     ''    using 2:14       ls 3 with l title 'NNLO', \
     ''    using 2:($6+$3+@egLL):($6+$3-@egLL)  ls 4 with filledcurves notitle, \
     ''    using 2:($6+$3)  ls 4 with l title 'LO+LL', \
     ''    using 2:($10+$4+@eg):($10+$4-@eg)  ls 5 with filledcurves notitle, \
     ''    using 2:($10+$4) ls 5 with l title 'NLO+NLL', \
     ''    using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 6 with filledcurves notitle, \
     ''    using 2:($14+$4-$32) ls 6 with l title 'NNLO+NLL', \
#     ''    using 2:($10+$38+@egLL):($10+$38-@egLL)  ls 7 with filledcurves notitle, \
#     ''    using 2:($10+$38)  ls 7 with l title 'NLO+LL', \
#     ''    using 2:(as/100.+$35):(as/100.+$36)  ls 7 with filledcurves notitle, \
#     ''    using 2:(as/100.+$35) ls 7 with l title 'as+logRdot', \
#     ''    using 2:($10+$3-$34)  ls 8 with l title 'NLO+LL', \
     #
#
set ylabel 'x P_{gq}(x)'
set yrange [0:0.01*as]
plot file  using 2:7           ls 1 with l title 'LO', \
     ''    using 2:11          ls 2 with l title 'NLO', \
     ''    using 2:15          ls 3 with l title 'NNLO', \
     ''    using 2:($7 +c*$3+c*@egLL):($7+c*$3-c*@egLL)  ls 4 with filledcurves notitle, \
     ''    using 2:($7 +c*$3)  ls 4 with l title 'LO+LL', \
     ''    using 2:($11+c*($4+@eg)):($11+c*($4-@eg))  ls 5 with filledcurves notitle, \
     ''    using 2:($11+c*$4)  ls 5 with l title 'NLO+NLL', \
     ''    using 2:($15+c*($4-$32+@eg)):($15+c*($4-$32-@eg))  ls 6 with filledcurves notitle, \
     ''    using 2:($15+c*($4-$32))  ls 6 with l title 'NNLO+NLL', \
#     ''    using 2:($11+c*($38+@egLL)):($11+c*($38-@egLL))  ls 7 with filledcurves notitle, \
#     ''    using 2:($11+c*$38)  ls 7 with l title 'NLO+LL', \
#     ''    using 2:($11+c*($3-$34))  ls 8 with l title 'NLO+LL', \
#
set ylabel 'x P_{qg}(x)'
set yrange [0:0.0125*as]
#set yrange [0:0.02]
plot file  using 2:8         ls 1 with l title 'LO', \
     ''    using 2:12        ls 2 with l title 'NLO', \
     ''    using 2:16        ls 3 with l title 'NNLO', \
     ''    using 2:($12+$5+@eq):($12+$5-@eq)  ls 5 with filledcurves notitle, \
     ''    using 2:($12+$5)  ls 5 with l title 'NLO+NLL', \
     ''    using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 6 with filledcurves notitle, \
     ''    using 2:($16+$5-$33)  ls 6 with l title 'NNLO+NLL', \
#
set ylabel 'x P_{qq}(x)'
set yrange [0:0.007*as]
plot file  using 2:9           ls 1 with l title 'LO', \
     ''    using 2:13          ls 2 with l title 'NLO', \
     ''    using 2:17          ls 3 with l title 'NNLO', \
     ''    using 2:($13+c*($5+@eq)):($13+c*($5-@eq))  ls 5 with filledcurves notitle, \
     ''    using 2:($13+c*$5)  ls 5 with l title 'NLO+NLL', \
     ''    using 2:($17+c*($5-$33+@eq)):($17+c*($5-$33-@eq))  ls 6 with filledcurves notitle, \
     ''    using 2:($17+c*($5-$33))  ls 6 with l title 'NNLO+NLL', \
#
#
#set terminal pdfcairo dashed font "Avenir, 10" enhanced lw 0.8 dl 1 size 3.1,2.3
#set terminal pdfcairo dashed font "Lucida grande, 10" enhanced lw 0.8 dl 1 size 3.1,2.3
set output 'plots/plot_P_nf4_paper_noband.pdf'
set key at graph 0.2,0.95
set ylabel 'x P_{gg}(x)'
as=20
set yrange [0:0.02*as]
#set yrange [0:0.1]
set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
file  = 'output/P_as0'.as.'_nf4.dat'
#
plot file  using 2:6        ls 1 with l title 'LO', \
     ''    using 2:10       ls 2 with l title 'NLO', \
     ''    using 2:14       ls 3 with l title 'NNLO', \
     ''    using 2:($6+$3)  ls 4 with l title 'LO+LLx', \
     ''    using 2:($10+$4) ls 5 with l title 'NLO+NLLx', \
     ''    using 2:($14+$4-$32) ls 6 with l title 'NNLO+NLLx', \
     #
#
set ylabel 'x P_{gq}(x)'
set yrange [0:0.01*as]
plot file  using 2:7           ls 1 with l title 'LO', \
     ''    using 2:11          ls 2 with l title 'NLO', \
     ''    using 2:15          ls 3 with l title 'NNLO', \
     ''    using 2:($7 +c*$3)  ls 4 with l title 'LO+LLx', \
     ''    using 2:($11+c*$4)  ls 5 with l title 'NLO+NLLx', \
     ''    using 2:($15+c*($4-$32))  ls 6 with l title 'NNLO+NLLx', \
#     ''    using 2:($11+c*($3-$34))  ls 8 with l title 'NLO+LL', \
#
set ylabel 'x P_{qg}(x)'
set yrange [0:0.0125*as]
#set yrange [0:0.02]
plot file  using 2:8         ls 1 with l title 'LO', \
     ''    using 2:12        ls 2 with l title 'NLO', \
     ''    using 2:16        ls 3 with l title 'NNLO', \
     ''    using 2:($12+$5)  ls 5 with l title 'NLO+NLLx', \
     ''    using 2:($16+$5-$33)  ls 6 with l title 'NNLO+NLLx', \
#
set ylabel 'x P_{qq}(x)'
set yrange [0:0.007*as]
plot file  using 2:9           ls 1 with l title 'LO', \
     ''    using 2:13          ls 2 with l title 'NLO', \
     ''    using 2:17          ls 3 with l title 'NNLO', \
     ''    using 2:($13+c*$5)  ls 5 with l title 'NLO+NLLx', \
     ''    using 2:($17+c*($5-$33))  ls 6 with l title 'NNLO+NLLx', \
#
#
set style lines 1 lc 0 lw 2 dt 3
set style lines 2 lc 0 lw 2 dt 2
set style lines 3 lc 0 lw 2 dt 5
set terminal pdfcairo dashed font "Palatino, 10" enhanced lw 0.8 dl 1 size 3.1,2.3
set output 'plots/plot_P_nf4_paper_fixedorder.pdf'
set key at graph 0.2,0.95
#set key at graph 0.3,0.95
set ylabel 'x P_{gg}(x)'
#set ylabel 'x γ_{gg}(x)'
as=20
set xrange [1:1e-9]
set xrange [1e-7:1] noreverse
set yrange [0:0.02*as]
#set yrange [0:0.1]
set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
file  = 'output/P_as0'.as.'_nf4.dat'
#
plot file  using 2:6        ls 1 with l title 'LO', \
     ''    using 2:10       ls 2 with l title 'NLO', \
     ''    using 2:14       ls 3 with l title 'NNLO', \
#
#
set ylabel 'x P_{gq}(x)'
set yrange [0:0.01*as]
plot file  using 2:7           ls 1 with l title 'LO', \
     ''    using 2:11          ls 2 with l title 'NLO', \
     ''    using 2:15          ls 3 with l title 'NNLO', \
#
set ylabel 'x P_{qg}(x)'
set yrange [0:0.0125*as]
#set yrange [0:0.02]
plot file  using 2:8         ls 1 with l title 'LO', \
     ''    using 2:12        ls 2 with l title 'NLO', \
     ''    using 2:16        ls 3 with l title 'NNLO', \
#
set ylabel 'x P_{qq}(x)'
set yrange [0:0.007*as]
plot file  using 2:9           ls 1 with l title 'LO', \
     ''    using 2:13          ls 2 with l title 'NLO', \
     ''    using 2:17          ls 3 with l title 'NNLO', \
#
set style lines 16 lc @b lw 2 dt 1
set ylabel 'x P_{gg}(x)'
#set ylabel 'x γ_{gg}(x)'
set yrange [0:0.02*as]
plot file  using 2:6        ls 1 with l title 'LO', \
     ''    using 2:10       ls 2 with l title 'NLO', \
     ''    using 2:14       ls 3 with l title 'NNLO', \
     ''    using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 16 with filledcurves notitle, \
     ''    using 2:($14+$4-$32) ls 16 with l title 'NNLO+NLLx', \
#
plot file  using 2:6        ls 1 with l title 'LO', \
     ''    using 2:10       ls 2 with l title 'NLO', \
     ''    using 2:14       ls 3 with l title 'NNLO', \
     ''    using 2:($14+$39)    ls 36 with l title 'N^3LO approx', \
#
plot file  using 2:6        ls 1 with l title 'LO', \
     ''    using 2:10       ls 2 with l title 'NLO', \
     ''    using 2:14       ls 3 with l title 'NNLO', \
     ''    using 2:($14+$39)    ls 36 with l title 'N^3LO approx', \
     ''    using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 16 with filledcurves notitle, \
     ''    using 2:($14+$4-$32) ls 16 with l title 'NNLO+NLLx', \
#
set ylabel 'x P_{qg}(x)'
set yrange [0:0.0125*as]
#set yrange [0:0.02]
plot file  using 2:8         ls 1 with l title 'LO', \
     ''    using 2:12        ls 2 with l title 'NLO', \
     ''    using 2:16        ls 3 with l title 'NNLO', \
     ''    using 2:($16+$40)    ls 36 with l title 'N^3LO approx', \
#
plot file  using 2:8         ls 1 with l title 'LO', \
     ''    using 2:12        ls 2 with l title 'NLO', \
     ''    using 2:16        ls 3 with l title 'NNLO', \
     ''    using 2:($16+$40)    ls 36 with l title 'N^3LO approx', \
     ''    using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 16 with filledcurves notitle, \
     ''    using 2:($16+$5-$33) ls 16 with l title 'NNLO+NLL', \
#
set key inside

#
set style lines 1 lc 0 lw 2 dt 3
set style lines 2 lc 0 lw 2 dt 2
set style lines 3 lc 0 lw 2 dt 5
set style lines 6 lc 0 lw 2 dt 1
set style lines 7 lc 0 lw 1 dt 4
set style lines 8 lc 0 lw 1 dt 1
#
set style lines 11 lc @b lw 2 dt 3
set style lines 12 lc @b lw 2 dt 2
set style lines 13 lc @b lw 2 dt 5
set style lines 14 lc @b lw 1 dt 4
set style lines 15 lc @b lw 1 dt 1
set style lines 16 lc @b lw 2 dt 1
#
set style lines 21 lc @r lw 2 dt 3
set style lines 22 lc @r lw 2 dt 2
set style lines 23 lc @r lw 2 dt 5
set style lines 24 lc @r lw 1 dt 4
set style lines 25 lc @r lw 1 dt 1
set style lines 26 lc @r lw 2 dt 1
#
set style lines 30 lc @w lw 0 dt 0
set style lines 31 lc @g lw 2 dt 3
set style lines 32 lc @g lw 2 dt 2
set style lines 33 lc @g lw 2 dt 5
set style lines 34 lc @g lw 1 dt 4
#set style lines 36 lc @g lw 2 dt 1
set style lines 36 lc @g lw 2 dt 4
#
as=28
set output 'plots/plot_P_nf4_as0'.as.'_mixed.pdf'
set key at graph 0.5,0.95
set key maxrows 4
set ylabel 'x P(x)'
set xrange [1:1e-5]
set yrange [0:0.02*as]
#set yrange [0:0.1]
#set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
set title 'α_s = 0.'.as.',  n_f = 4'
file  = 'output/P_as0'.as.'_nf4.dat'
#
plot file  using 2:(1/0)        ls 1 with l title 'LO', \
     ''    using 2:(1/0)        ls 2 with l title 'NLO          ', \
     ''    using 2:(1/0)        ls 3 with l title 'NNLO', \
     ''    using 2:(1/0)        ls 6 with l title 'NNLO+NLLx', \
     ''    using 2:6            ls 11 with l notitle 'LO', \
     ''    using 2:10           ls 12 with l notitle 'NLO', \
     ''    using 2:14           ls 13 with l notitle 'NNLO', \
     ''    using 2:($14+$4-$32) ls 16 with l title 'P_{gg}', \
     ''    using 2:8            ls 21 with l notitle 'LO', \
     ''    using 2:12           ls 22 with l notitle 'NLO', \
     ''    using 2:16           ls 23 with l notitle 'NNLO', \
     ''    using 2:($16+$5-$33) ls 26 with l title 'P_{qg}', \
     ''    using 2:(1/0)        ls 30 with l title ' '
#
#



as=20
set output 'plots/plot_P_nf4_as0'.as.'_mixed.pdf'
#set key at graph 0.5,0.95
set key at graph 0.7,0.95
set key maxrows 4
set ylabel 'x P_{ij}(x)'
#set xrange [1:1e-9]
set xrange [1e-7:1]
#set yrange [0:0.1]
set xzeroaxis lt -1 lw 0.1 lc @gr
set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
#set title 'α_s = 0.'.as.',  n_f = 4'
file   = 'output/P_as0'.as.'_nf4.dat'
fileL  = 'output/P_as0'.as.'_nf4_LLp.dat'
file2L = 'output/P_as0'.as.'_nf4_LLp.HELL2bug.dat'
#
set yrange [0:0.02*as]
#set yrange [-0.2:0.5]
plot file  using 2:(1/0)        ls 1 with l title 'LO', \
     ''    using 2:(1/0)        ls 2 with l title 'NLO          ', \
     ''    using 2:(1/0)        ls 3 with l title 'NNLO', \
     ''    using 2:(1/0)        ls 6 with l title 'NNLO+NLL', \
     ''    using 2:6            ls 11 with l notitle 'LO', \
     ''    using 2:10           ls 12 with l notitle 'NLO', \
     ''    using 2:14           ls 13 with l notitle 'NNLO', \
     ''    using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 16 with filledcurves notitle, \
     ''    using 2:($14+$4-$32) ls 16 with l title 'P_{gg}', \
     ''    using 2:8            ls 21 with l notitle 'LO', \
     ''    using 2:12           ls 22 with l notitle 'NLO', \
     ''    using 2:16           ls 23 with l notitle 'NNLO', \
     ''    using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 26 with filledcurves notitle, \
     ''    using 2:($16+$5-$33) ls 26 with l title 'P_{qg}', \
     ''    using 2:(1/0)        ls 30 with l title ' ', \
#     ''    using 2:($14+$39)    ls 14 with l notitle 'N3LOapprox', \
#     ''    using 2:($16+$40)    ls 24 with l notitle 'N3LOapprox', \
#     ''    using 2:($14+(as/100.)**4*(cgg34*log($2)**3+cgg33Q0(4)*log($2)**2-0*50*log($2))) ls 11 with l notitle, \
#
#
set yrange [0:0.01*as]
plot file  using 2:(1/0)        ls 1 with l title 'LO', \
     ''    using 2:(1/0)        ls 2 with l title 'NLO          ', \
     ''    using 2:(1/0)        ls 3 with l title 'NNLO', \
     ''    using 2:(1/0)        ls 6 with l title 'NNLO+NLL', \
     ''    using 2:7            ls 11 with l notitle 'LO', \
     ''    using 2:11           ls 12 with l notitle 'NLO', \
     ''    using 2:15           ls 13 with l notitle 'NNLO', \
     ''    using 2:($15+c*($4-$32+@eg)):($15+c*($4-$32-@eg))  ls 16 with filledcurves notitle, \
     ''    using 2:($15+c*($4-$32)) ls 16 with l title 'P_{gq}', \
     ''    using 2:9            ls 21 with l notitle 'LO', \
     ''    using 2:13           ls 22 with l notitle 'NLO', \
     ''    using 2:17           ls 23 with l notitle 'NNLO', \
     ''    using 2:($17+c*($5-$33+@eq)):($17+c*($5-$33-@eq))  ls 26 with filledcurves notitle, \
     ''    using 2:($17+c*($5-$33)) ls 26 with l title 'P_{qq}', \
     ''    using 2:(1/0)        ls 30 with l title ' '
#
#
# set key maxrows 6
# set yrange [0:0.02*as]
# #set yrange [-0.2:0.5]
# plot file  using 2:(1/0)        ls 1 with l title 'LO', \
#      ''    using 2:(1/0)        ls 2 with l title 'NLO          ', \
#      ''    using 2:(1/0)        ls 3 with l title 'NNLO', \
#      ''    using 2:(1/0)        ls 6 with l title 'NNLO+NLL', \
#      ''    using 2:(1/0)        ls 7 with l title 'N^3LO approx', \
#      ''    using 2:6            ls 11 with l notitle 'LO', \
#      ''    using 2:10           ls 12 with l notitle 'NLO', \
#      ''    using 2:14           ls 13 with l notitle 'NNLO', \
#      ''    using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 16 with filledcurves notitle, \
#      ''    using 2:($14+$4-$32) ls 16 with l title 'P_{gg}', \
#      ''    using 2:8            ls 21 with l notitle 'LO', \
#      ''    using 2:12           ls 22 with l notitle 'NLO', \
#      ''    using 2:16           ls 23 with l notitle 'NNLO', \
#      ''    using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 26 with filledcurves notitle, \
#      ''    using 2:($16+$5-$33) ls 26 with l title 'P_{qg}', \
#      ''    using 2:(1/0)        ls 30 with l title ' ', \
#      ''    using 2:(1/0)        ls 30 with l title ' ', \
#      ''    using 2:(1/0)        ls 30 with l title ' ', \
#      ''    using 2:(1/0)        ls 30 with l title ' ', \
#      ''    using 2:($14+$39)    ls 14 with l notitle 'N^3LO approx', \
#      ''    using 2:($16+$40)    ls 24 with l notitle 'N^3LO approx', \
# #     ''    using 2:(1/0)        ls 8 with l title 'N^3LO asympt', \
# #     ''    using 2:($14+$39+$41):($14+$39-$41)  ls 14 with filledcurves notitle, \
# #     ''    using 2:($2>0.001?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33Q0(4)*log($2)**2)) ls 15 with l notitle 'N^3LO asympt', \
# #     ''    using 2:($16+$40+$42):($16+$40-$42)  ls 24 with filledcurves notitle, \
# #     ''    using 2:($2>0.001?0/0:$16+(as/100.)**4*(cqg33(4)*log($2)**2)) ls 25 with l notitle 'N^3LO asympt', \
# #     ''    using 2:($2>0.001?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33MS(4)*log($2)**2)) ls 14 with l title 'N^3LO asympt ~{MS}{‾‾‾}', \
# #     ''    using 2:($14+$39)    ls 14 with l notitle 'N3LOapprox', \
# #     ''    using 2:($16+$40)    ls 24 with l notitle 'N3LOapprox', \
# #     ''    using 2:($14+(as/100.)**4*(cgg34*log($2)**3+cgg33Q0(4)*log($2)**2-0*50*log($2))) ls 11 with l notitle, \
# #
#
set style lines 11 lc @bl lw 1 dt 3
set style lines 12 lc @bl lw 1 dt 2
set style lines 13 lc @bl lw 1 dt 5
#
set style lines 21 lc @bl lw 1 dt 3
set style lines 22 lc @bl lw 1 dt 2
set style lines 23 lc @bl lw 1 dt 5
#
set style lines 24 lc 0   lw 1 dt 1
set style lines 26 lc @re lw 2 dt 1
set style lines 15 lc @r lw 1 dt 2
set style lines 25 lc @r lw 1 dt 1
#
set style lines 46 lc @gr lw 1 dt 4
#
set key maxrows 8
set key inside top left
set yrange [0:0.02*as]
set yrange [0:0.015*as]
set ylabel 'x P_{qg}(x)'
plot file   using 2:8            ls 21 with l title 'LO', \
     ''     using 2:12           ls 22 with l title 'NLO', \
     ''     using 2:16           ls 23 with l title 'NNLO', \
     ''     using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 26 with filledcurves notitle, \
     ''     using 2:($16+$5-$33)                       ls 26 with l title 'NNLO+NLL', \
     fileL  using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 36 with filledcurves notitle, \
     fileL  using 2:($16+$5-$33)                       ls 36 with l title "NNLO+NLL (LL{\047})", \
#
set key at graph 0.1,0.95
set yrange [0:0.02*as]
set ylabel 'x P_{gg}(x)'
plot file   using 2:6            ls 11 with l title 'LO', \
     ''     using 2:10           ls 12 with l title 'NLO', \
     ''     using 2:14           ls 13 with l title 'NNLO', \
     ''     using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 26 with filledcurves notitle, \
     ''     using 2:($14+$4-$32)                       ls 26 with l title 'NNLO+NLL', \
     fileL  using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 36 with filledcurves notitle, \
     ''     using 2:($14+$4-$32)                       ls 36 with l title "NNLO+NLL (LL{\047})", \
#
set key inside top left
set yrange [0:0.02*as]
set yrange [0:0.015*as]
set ylabel 'x P_{qg}(x)'
plot file   using 2:8            ls 21 with l title 'LO', \
     ''     using 2:12           ls 22 with l title 'NLO', \
     ''     using 2:16           ls 23 with l title 'NNLO', \
     ''     using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 26 with filledcurves notitle, \
     ''     using 2:($16+$5-$33)                       ls 26 with l title 'NNLO+NLL', \
     fileL  using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 36 with filledcurves notitle, \
     fileL  using 2:($16+$5-$33)                       ls 36 with l title "NNLO+NLL (LL{\047})", \
     file2L using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 46 with filledcurves fs transparent pattern 6 notitle, \
     ''     using 2:($16+$5-$33)                       ls 46 with l title "NNLO+NLL (HELL 2.0)", \
#
set key at graph 0.1,0.95
set yrange [0:0.02*as]
set ylabel 'x P_{gg}(x)'
plot file   using 2:6            ls 11 with l title 'LO', \
     ''     using 2:10           ls 12 with l title 'NLO', \
     ''     using 2:14           ls 13 with l title 'NNLO', \
     ''     using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 26 with filledcurves notitle, \
     ''     using 2:($14+$4-$32)                       ls 26 with l title 'NNLO+NLL', \
     fileL  using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 36 with filledcurves notitle, \
     ''     using 2:($14+$4-$32)                       ls 36 with l title "NNLO+NLL (LL{\047})", \
     file2L using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 46 with filledcurves fs transparent pattern 6 notitle, \
     ''     using 2:($14+$4-$32)                       ls 46 with l title "NNLO+NLL (HELL 2.0)", \
#
set key inside top left
set yrange [0:0.02*as]
set yrange [0:0.015*as]
set ylabel 'x P_{qg}(x)'
plot file   using 2:8            ls 21 with l title 'LO', \
     ''     using 2:12           ls 22 with l title 'NLO', \
     ''     using 2:16           ls 23 with l title 'NNLO', \
     fileL  using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 36 with filledcurves notitle, \
     fileL  using 2:($16+$5-$33)                       ls 36 with l title "NNLO+NLL (LL{\047})", \
     file2L using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 46 with filledcurves fs transparent pattern 6 notitle, \
     ''     using 2:($16+$5-$33)                       ls 46 with l title "NNLO+NLL (HELL 2.0)", \
#
set key at graph 0.1,0.95
set yrange [0:0.02*as]
set ylabel 'x P_{gg}(x)'
plot file   using 2:6            ls 11 with l title 'LO', \
     ''     using 2:10           ls 12 with l title 'NLO', \
     ''     using 2:14           ls 13 with l title 'NNLO', \
     fileL  using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 36 with filledcurves notitle, \
     ''     using 2:($14+$4-$32)                       ls 36 with l title "NNLO+NLL (LL{\047})", \
     file2L using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 46 with filledcurves fs transparent pattern 6 notitle, \
     ''     using 2:($14+$4-$32)                       ls 46 with l title "NNLO+NLL (HELL 2.0)", \
#
#
#
set key inside top left
set yrange [-0.004*as:0.004*as]
set ylabel 'x ∆_4P_{qg}(x)'
plot file  using 2:($44+$46):($44-$46)  ls 26 with filledcurves notitle, \
     ''    using 2:44                   ls 26 with l title 'resummed contribution to N^3LO', \
     fileL using 2:($44+$46):($44-$46)  ls 36 with filledcurves notitle, \
     ''    using 2:44                   ls 36 with l title "resummed contribution to N^3LO (LL')", \
#
set yrange [-0.01*as:0.02*as]
set ylabel 'x ∆_4P_{gg}(x)'
plot file  using 2:($43+$45):($43-$45)  ls 26 with filledcurves notitle, \
     ''    using 2:43                   ls 26 with l title 'resummed contribution to N^3LO', \
     fileL using 2:($43+$45):($43-$45)  ls 36 with filledcurves notitle, \
     ''    using 2:43                   ls 36 with l title "resummed contribution to N^3LO (LL')", \
#
#
#
#
#
set key inside top left
set xrange [1:1e-9] reverse
set yrange [-0.002*as:0.015*as]
set ylabel 'x P_{qg}(x)'
plot file  using 2:8            ls 21 with l title 'LO', \
     ''    using 2:12           ls 22 with l title 'NLO', \
     ''    using 2:16           ls 23 with l title 'NNLO', \
     ''    using 2:($16+$40+$42):($16+$40-$42)  ls 26 with filledcurves notitle, \
     ''    using 2:($16+$40)    ls 26 with l title 'N^3LO approx', \
     fileL using 2:($16+$40+$42):($16+$40-$42)  ls 36 with filledcurves notitle, \
     fileL using 2:($16+$40)    ls 36 with l title "N^3LO approx (LL')", \
     ''    using 2:($2>0.001?0/0:$16+(as/100.)**4*(cqg33(4)*log($2)**2)) ls 24 with l title 'N^3LO asympt', \
#     ''    using 2:($2>0.001?0/0:$12+(as/100.)**3*(cqg22(4)*log($2))) ls 24 with l title 'NNLO asympt', \
#
set key maxrows 3
set key inside top center
set yrange [-0.01*as:0.02*as]
set ylabel 'x P_{gg}(x)'
plot file  using 2:6            ls 11 with l title 'LO', \
     ''    using 2:10           ls 12 with l title 'NLO', \
     ''    using 2:14           ls 13 with l title 'NNLO', \
     ''    using 2:($14+$39+$41):($14+$39-$41)  ls 26 with filledcurves notitle, \
     ''    using 2:($14+$39)    ls 26 with l title 'N^3LO approx', \
     fileL using 2:($14+$39+$41):($14+$39-$41)  ls 36 with filledcurves notitle, \
     fileL using 2:($14+$39)    ls 36 with l title "N^3LO approx (LL')", \
     ''    using 2:($2>0.01?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33Q0(4)*log($2)**2)) ls 24 with l title 'N^3LO asympt', \
#     ''    using 2:($2>0.001?0/0:$10+(as/100.)**3*(cgg22(4)*log($2))) ls 24 with l title 'NNLO asympt', \
#
set key maxrows 8
set key inside bottom left
set yrange [-0.04*as:0.03*as]
plot file  using 2:6            ls 11 with l title 'LO', \
     ''    using 2:10           ls 12 with l title 'NLO', \
     ''    using 2:14           ls 13 with l title 'NNLO', \
     ''    using 2:($14+$39+$41):($14+$39-$41)  ls 26 with filledcurves notitle, \
     ''    using 2:($14+$39)    ls 26 with l title 'N^3LO approx', \
     fileL using 2:($14+$39+$41):($14+$39-$41)  ls 36 with filledcurves notitle, \
     fileL using 2:($14+$39)    ls 36 with l title "N^3LO approx (LL')", \
     ''    using 2:($2>0.001?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33Q0(4)*log($2)**2)) ls 24 with l title 'N^3LO asympt', \
     ''    using 2:($2>0.001?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33MS(4)*log($2)**2)) ls 14 with l title 'N^3LO asympt ~{MS}{‾‾‾}', \
#     ''    using 2:($2>0.001?0/0:$10+(as/100.)**3*(cgg22(4)*log($2))) ls 24 with l title 'NNLO asympt', \
#
set multiplot
set yrange [-0.04*as:0.04*as]
replot
set origin 0.11,0.55
set size 0.516,0.37
set xrange [0.1:1e-5]
set yrange [0.015:0.205]
set notitle
set nokey
set noxlabel
set noylabel
set xtics offset 100
set ytics 0.04 left offset 29
replot
unset multiplot
#reset
set origin 0,0
set size 1,1
set ytics auto right offset 0
set xtics offset 0
set xlabel 'x'



set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
set key inside top left
set xrange [1:1e-9]
set yrange [-0.002*as:0.015*as]
set ylabel 'x P_{qg}(x)'
plot file  using 2:8            ls 21 with l title 'LO', \
     ''    using 2:12           ls 22 with l title 'NLO', \
     ''    using 2:16           ls 23 with l title 'NNLO', \
     ''    using 2:($16+$5-$33+@eq):($16+$5-$33-@eq)  ls 24 with filledcurves notitle, \
     ''    using 2:($16+$5-$33) ls 24 with l title 'NNLO+NLL', \
     ''    using 2:($16+$40)    ls 26 with l title 'N^3LO approx', \
     ''    using 2:($2>0.001?0/0:$16+(as/100.)**4*(cqg33(4)*log($2)**2)) ls 25 with l title 'N^3LO asympt', \
#     ''    using 2:($2>0.001?0/0:$12+(as/100.)**3*(cqg22(4)*log($2))) ls 24 with l title 'NNLO asympt', \
#
set key maxrows 8
set key inside bottom left
#set multiplot
set yrange [-0.04*as:0.03*as]
plot file  using 2:6            ls 11 with l title 'LO', \
     ''    using 2:10           ls 12 with l title 'NLO', \
     ''    using 2:14           ls 13 with l title 'NNLO', \
     ''    using 2:($14+$4-$32+@eg):($14+$4-$32-@eg)  ls 24 with filledcurves notitle, \
     ''    using 2:($14+$4-$32) ls 24 with l title 'NNLO+NLL', \
     ''    using 2:($14+$39+$41):($14+$39-$41)  ls 26 with filledcurves notitle, \
     ''    using 2:($14+$39)    ls 26 with l title 'N^3LO approx', \
     ''    using 2:($2>0.001?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33Q0(4)*log($2)**2)) ls 25 with l title 'N^3LO asympt', \
     ''    using 2:($2>0.001?0/0:$14+(as/100.)**4*(cgg34*log($2)**3+cgg33MS(4)*log($2)**2)) ls 15 with l title 'N^3LO asympt ~{MS}{‾‾‾}', \
#     ''    using 2:($2>0.001?0/0:$10+(as/100.)**3*(cgg22(4)*log($2))) ls 24 with l title 'NNLO asympt', \
#
# set origin 0.11,0.55
# set size 0.516,0.37
# set xrange [0.1:1e-5]
# set yrange [0.015:0.205]
# set notitle
# set nokey
# set noxlabel
# set noylabel
# set xtics offset 100
# set ytics 0.04 left offset 29
# replot
# unset multiplot
# #reset
# set origin 0,0
# set size 1,1
# set ytics auto right offset 0
# set xtics offset 0
# set xlabel 'x'
# 





bl='rgbcolor "#cccccc"'
set style lines 1 lc @bl lw 1 dt 3
set style lines 2 lc @bl lw 1 dt 2
set style lines 3 lc @bl lw 1 dt 5

set style lines 4 lc @g  lw 2 dt 3
set style lines 5 lc @b  lw 2 dt 1
set style lines 8 lc @b  lw 2 dt 1
set style lines 9 lc @g  lw 2 dt 4
set style lines 6 lc @p  lw 2 dt 4
set style lines 77 lc @y  lw 2 dt 2
set style lines  7 lc @dy lw 2 dt 2

set style fill transparent solid 0.2 noborder

set terminal pdfcairo dashed font "Palatino, 12" enhanced lw 0.8 dl 1 size 3.5,2.2
set key top left maxrows 6
as=20
set output 'plots/plot_P_nf4_as0'.as.'_paper_ratio.pdf'
set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
file  = 'output/P_as0'.as.'_nf4.dat'
fileL = 'output/P_as0'.as.'_nf4_LLp.dat'
set xrange [1:1e-7]
set yrange [0:2.5]
set ylabel 'P_{gg}(x)    (ratio to NLO)'
plot file  using 2:($6/$10)        ls 1 with l title 'LO', \
     ''    using 2:(1)             ls 2 with l title 'NLO', \
     ''    using 2:($14/$10)       ls 3 with l title 'NNLO', \
     ''    using 2:(($14+$4-$32+@eg)/$10):(($14+$4-$32-@eg)/$10)  ls 8 with filledcurves notitle, \
     fileL using 2:(($14+$4-$32+@eg)/$10):(($14+$4-$32-@eg)/$10)  ls 9 with filledcurves notitle, \
     file  using 2:(($14+$4-$32)/$10)  ls 8 with l title 'NNLO+NLL (HELL 3.0)', \
     fileL using 2:(($14+$4-$32)/$10)  ls 9 with l title 'NNLO+NLL (HELL 2.0)', \
#
#
set ylabel 'P_{gq}(x)    (ratio to NLO)'
plot file  using 2:($7/$11)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($15/$11)          ls 3 with l title 'NNLO', \
     ''    using 2:(($15+c*($4-$32+@eg))/$11):(($15+c*($4-$32-@eg))/$11)  ls 8 with filledcurves notitle, \
     fileL using 2:(($15+c*($4-$32+@eg))/$11):(($15+c*($4-$32-@eg))/$11)  ls 9 with filledcurves notitle, \
     file  using 2:(($15+c*($4-$32))/$11)   ls 8 with l title 'NNLO+NLL (HELL 3.0)', \
     fileL using 2:(($15+c*($4-$32))/$11)   ls 9 with l title 'NNLO+NLL (HELL 2.0)', \
     #
#
set yrange [0:4]
set ylabel 'P_{qg}(x)    (ratio to NLO)'
plot file  using 2:($8/$12)         ls 1 with l title 'LO', \
     ''    using 2:($12/$12)        ls 2 with l title 'NLO', \
     ''    using 2:($16/$12)        ls 3 with l title 'NNLO', \
     ''    using 2:(($16+$5-$33+@eq)/$12):(($16+$5-$33-@eq)/$12)  ls 8 with filledcurves notitle, \
     fileL using 2:(($16+$5-$33+@eq)/$12):(($16+$5-$33-@eq)/$12)  ls 9 with filledcurves notitle, \
     file  using 2:(($16+$5-$33)/$12)   ls 8 with l title 'NNLO+NLL (HELL 3.0)', \
     fileL using 2:(($16+$5-$33)/$12)   ls 9 with l title 'NNLO+NLL (HELL 2.0)', \
#
set ylabel 'P_{qq}(x)    (ratio to NLO)'
plot file  using 2:($9/$13)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($17/$13)          ls 3 with l title 'NNLO', \
     ''    using 2:(($17+c*($5-$33+@eq))/$13):(($17+c*($5-$33-@eq))/$13)  ls 8 with filledcurves notitle, \
     fileL using 2:(($17+c*($5-$33+@eq))/$13):(($17+c*($5-$33-@eq))/$13)  ls 9 with filledcurves notitle, \
     file  using 2:(($17+c*($5-$33))/$13)   ls 8 with l title 'NNLO+NLL (HELL 3.0)', \
     fileL using 2:(($17+c*($5-$33))/$13)   ls 9 with l title 'NNLO+NLL (HELL 2.0)', \
#
#


set key top left maxrows 6
set yrange [0:2.5]
set ylabel 'P_{gg}(x)    (ratio to NLO)'
plot file  using 2:($6/$10)        ls 1 with l title 'LO', \
     ''    using 2:(1)             ls 2 with l title 'NLO', \
     ''    using 2:($14/$10)       ls 3 with l title 'NNLO', \
     ''    using 2:(($10+$4+@eg)/$10):(($10+$4-@eg)/$10)  ls 5 with filledcurves notitle, \
     fileL using 2:(($10+$4+@eg)/$10):(($10+$4-@eg)/$10)  ls 9 with filledcurves notitle, \
     file  using 2:(($10+$4)/$10)  ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     fileL using 2:(($10+$4)/$10)  ls 9 with l title 'NLO+NLL (HELL 2.0)', \
#
#
set ylabel 'P_{gq}(x)    (ratio to NLO)'
plot file  using 2:($7/$11)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($15/$11)          ls 3 with l title 'NNLO', \
     file  using 2:(($11+c*($4+@eg))/$11):(($11+c*($4-@eg))/$11)  ls 5 with filledcurves notitle, \
     fileL using 2:(($11+c*($4+@eg))/$11):(($11+c*($4-@eg))/$11)  ls 9 with filledcurves notitle, \
     file  using 2:(($11+c*$4)/$11)   ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     fileL using 2:(($11+c*$4)/$11)   ls 9 with l title 'NLO+NLL (HELL 2.0)', \
#
set key top left maxrows 6
set yrange [0:4]
set ylabel 'P_{qg}(x)    (ratio to NLO)'
plot file  using 2:($8/$12)         ls 1 with l title 'LO', \
     ''    using 2:($12/$12)        ls 2 with l title 'NLO', \
     ''    using 2:($16/$12)        ls 3 with l title 'NNLO', \
     file  using 2:(($12+$5+@eq)/$12):(($12+$5-@eq)/$12)  ls 5 with filledcurves notitle, \
     fileL using 2:(($12+$5+@eq)/$12):(($12+$5-@eq)/$12)  ls 9 with filledcurves notitle, \
     file  using 2:(($12+$5)/$12)   ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     fileL using 2:(($12+$5)/$12)   ls 9 with l title 'NLO+NLL (HELL 2.0)', \
#
set ylabel 'P_{qq}(x)    (ratio to NLO)'
plot file  using 2:($9/$13)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($17/$13)          ls 3 with l title 'NNLO', \
     file  using 2:(($13+c*($5+@eq))/$13):(($13+c*($5-@eq))/$13)  ls 5 with filledcurves notitle, \
     fileL using 2:(($13+c*($5+@eq))/$13):(($13+c*($5-@eq))/$13)  ls 9 with filledcurves notitle, \
     file  using 2:(($13+c*$5)/$13)   ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     fileL using 2:(($13+c*$5)/$13)   ls 9 with l title 'NLO+NLL (HELL 2.0)', \
#
#


set yrange [0:2.5]
set ylabel 'P_{gg}(x)    (ratio to NLO)'
plot file  using 2:($6/$10)        ls 1 with l title 'LO', \
     ''    using 2:(1)             ls 2 with l title 'NLO', \
     ''    using 2:($14/$10)       ls 3 with l title 'NNLO', \
     ''    using 2:($19/$10):($20/$10) ls 77 with filledcurves notitle, \
     ''    using 2:(($10+$4+@eg)/$10):(($10+$4-@eg)/$10)  ls 5 with filledcurves notitle, \
     ''    using 2:(($10+$4)/$10)  ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     ABFf  using (1/$1):($3/$2)    ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($18/$10)       ls 7 with l title 'NLO+NLL (CCSS)', \
#     fileL using 2:(($10+$4+@eg)/$10):(($10+$4-@eg)/$10)  ls 9 with filledcurves notitle, \
#     ''    using 2:(($10+$4)/$10)  ls 9 with l title 'NLO+NLL (HELL 3.0)', \
#
#
set ylabel 'P_{gq}(x)    (ratio to NLO)'
plot file  using 2:($7/$11)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($15/$11)          ls 3 with l title 'NNLO', \
     ''    using 2:($22/$11):($23/$11)  ls 77 with filledcurves notitle, \
     ''    using 2:(($11+c*($4+@eg))/$11):(($11+c*($4-@eg))/$11)  ls 5 with filledcurves notitle, \
     ''    using 2:(($11+c*$4)/$11)   ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     ABFf  using (1/$1):($6/$8)       ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($21/$11)          ls 7 with l title 'NLO+NLL (CCSS)', \
#     ''    using 2:(($7 +c*$3)/$11)   ls 4 with l title 'LO+LL (this work)', \
     #
set key top left maxrows 6
set yrange [0:4]
set ylabel 'P_{qg}(x)    (ratio to NLO)'
plot file  using 2:($8/$12)         ls 1 with l title 'LO', \
     ''    using 2:($12/$12)        ls 2 with l title 'NLO', \
     ''    using 2:($16/$12)        ls 3 with l title 'NNLO', \
     ''    using 2:($25/$12):($26/$12) ls 77 with filledcurves notitle, \
     ''    using 2:(($12+$5+@eq)/$12):(($12+$5-@eq)/$12)  ls 5 with filledcurves notitle, \
     ''    using 2:(($12+$5)/$12)   ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     ABFf  using (1/$1):($5/$4)     ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($24/$12)        ls 7 with l title 'NLO+NLL (CCSS)', \
#
set ylabel 'P_{qq}(x)    (ratio to NLO)'
plot file  using 2:($9/$13)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($17/$13)          ls 3 with l title 'NNLO', \
     ''    using 2:($28/$13):($29/$13) ls 77 with filledcurves notitle, \
     ''    using 2:(($13+c*($5+@eq))/$13):(($13+c*($5-@eq))/$13)  ls 5 with filledcurves notitle, \
     ''    using 2:(($13+c*$5)/$13)   ls 5 with l title 'NLO+NLL (HELL 3.0)', \
     ABFf  using (1/$1):($7/$9)       ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($27/$13)          ls 7 with l title 'NLO+NLL (CCSS)', \
#
#








bl='rgbcolor "#cccccc"'
set style lines 1 lc @bl lw 1 dt 2
set style lines 2 lc @bl lw 1 dt 1
set style lines 3 lc @bl lw 1 dt 5

set style lines 4 lc @g  lw 2 dt 3
set style lines 5 lc @p  lw 2 dt 1
set style lines 8 lc @b  lw 2 dt 5
set style lines 6 lc @g  lw 2 dt 4
set style lines 77 lc @y  lw 2 dt 2
set style lines  7 lc @dy lw 2 dt 2


set terminal pdfcairo dashed font "Palatino, 12" enhanced lw 0.8 dl 1 size 3.5,2.2
set output 'plots/plot_P_nf4_paper_ratio.pdf'
set key top left maxrows 4
as=20
set title 'α_s = 0.'.as.',  n_f = 4,  Q_0~{MS}{‾‾‾}'
file  = 'output/P_as0'.as.'_nf4.dat'
set xrange [1:1e-7]
set yrange [0:2.5]
set ylabel 'P_{gg}(x)    (ratio to NLO)'
plot file  using 2:($6/$10)        ls 1 with l title 'LO', \
     ''    using 2:(1)             ls 2 with l title 'NLO', \
     ''    using 2:($14/$10)       ls 3 with l title 'NNLO', \
     ''    using 2:($19/$10):($20/$10) ls 77 with filledcurves notitle, \
     ''    using 2:(($10+$4+@eg)/$10):(($10+$4-@eg)/$10)  ls 5 with filledcurves notitle, \
     ''    using 2:(($14+$4-$32+@eg)/$10):(($14+$4-$32-@eg)/$10)  ls 8 with filledcurves notitle, \
     ''    using 2:(($14+$4-$32)/$10)  ls 8 with l title 'NNLO+NLL (this work)', \
     ''    using 2:(($10+$4)/$10)  ls 5 with l title 'NLO+NLL (this work)', \
     ABFf  using (1/$1):($3/$2)    ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($18/$10)       ls 7 with l title 'NLO+NLL (CCSS)', \
#     ''    using 2:(($6+$3)/$10)   ls 4 with l title 'LO+LL (this work)', \
     #
#
set ylabel 'P_{gq}(x)    (ratio to NLO)'
plot file  using 2:($7/$11)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($15/$11)          ls 3 with l title 'NNLO', \
     ''    using 2:($22/$11):($23/$11)  ls 77 with filledcurves notitle, \
     ''    using 2:(($11+c*($4+@eg))/$11):(($11+c*($4-@eg))/$11)  ls 5 with filledcurves notitle, \
     ''    using 2:(($15+c*($4-$32+@eg))/$11):(($15+c*($4-$32-@eg))/$11)  ls 8 with filledcurves notitle, \
     ''    using 2:(($15+c*($4-$32))/$11)   ls 8 with l title 'NNLO+NLL (this work)', \
     ''    using 2:(($11+c*$4)/$11)   ls 5 with l title 'NLO+NLL (this work)', \
     ABFf  using (1/$1):($6/$8)       ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($21/$11)          ls 7 with l title 'NLO+NLL (CCSS)', \
#     ''    using 2:(($7 +c*$3)/$11)   ls 4 with l title 'LO+LL (this work)', \
     #
set key top left maxrows 6
set yrange [0:4]
set ylabel 'P_{qg}(x)    (ratio to NLO)'
plot file  using 2:($8/$12)         ls 1 with l title 'LO', \
     ''    using 2:($12/$12)        ls 2 with l title 'NLO', \
     ''    using 2:($16/$12)        ls 3 with l title 'NNLO', \
     ''    using 2:($25/$12):($26/$12) ls 77 with filledcurves notitle, \
     ''    using 2:(($12+$5+@eq)/$12):(($12+$5-@eq)/$12)  ls 5 with filledcurves notitle, \
     ''    using 2:(($16+$5-$33+@eq)/$12):(($16+$5-$33-@eq)/$12)  ls 8 with filledcurves notitle, \
     ''    using 2:(($16+$5-$33)/$12)   ls 8 with l title 'NNLO+NLL (this work)', \
     ''    using 2:(($12+$5)/$12)   ls 5 with l title 'NLO+NLL (this work)', \
     ABFf  using (1/$1):($5/$4)     ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($24/$12)        ls 7 with l title 'NLO+NLL (CCSS)', \
#
set ylabel 'P_{qq}(x)    (ratio to NLO)'
plot file  using 2:($9/$13)           ls 1 with l title 'LO', \
     ''    using 2:(1)                ls 2 with l title 'NLO', \
     ''    using 2:($17/$13)          ls 3 with l title 'NNLO', \
     ''    using 2:($28/$13):($29/$13) ls 77 with filledcurves notitle, \
     ''    using 2:(($13+c*($5+@eq))/$13):(($13+c*($5-@eq))/$13)  ls 5 with filledcurves notitle, \
     ''    using 2:(($17+c*($5-$33+@eq))/$13):(($17+c*($5-$33-@eq))/$13)  ls 8 with filledcurves notitle, \
     ''    using 2:(($17+c*($5-$33))/$13)   ls 8 with l title 'NNLO+NLL (this work)', \
     ''    using 2:(($13+c*$5)/$13)   ls 5 with l title 'NLO+NLL (this work)', \
     ABFf  using (1/$1):($7/$9)       ls 6 with l title 'NLO+NLL (ABF)', \
     file  using 2:($27/$13)          ls 7 with l title 'NLO+NLL (CCSS)', \
#
#

bl='rgbcolor "#888888"'
set style lines 1 lc @bl lw 1 dt 2
set style lines 2 lc @bl lw 1 dt 1
set style lines 3 lc @bl lw 1 dt 5

set key top left maxrows 3
set yrange [0:2]
set ylabel 'P_{gg}(x)    (ratio to LO)'
plot file  using 2:($6/$6)        ls 1 with l title 'LO', \
     ''    using 2:($10/$6)       ls 2 with l title 'NLO', \
     ''    using 2:($14/$6)       ls 3 with l title 'NNLO', \
     ''    using 2:(($6+$3)/$6)   ls 5 with l title 'LO+LL (this work)', \
     ABFf1 using (1/$1):($2/$4)   ls 6 with l title 'LO+LL (ABF)', \
     #
#
set ylabel 'P_{gq}(x)    (ratio to LO)'
plot file  using 2:($7/$7)           ls 1 with l title 'LO', \
     ''    using 2:($11/$7)          ls 2 with l title 'NLO', \
     ''    using 2:($15/$7)          ls 3 with l title 'NNLO', \
     ''    using 2:(($7 +c*$3)/$7)   ls 5 with l title 'LO+LL (this work)', \
     ABFf2 using (1/$1):($2/$4)      ls 6 with l title 'LO+LL (ABF)', \
     #
#
#



quit

