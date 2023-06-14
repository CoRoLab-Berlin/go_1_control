import jax.numpy as jnp
import numpy as np
import jax
@jax.jit
def gravity_vetor(q):
	var1=jnp.cos(q[3])
	var2=jnp.cos(q[4])
	var3=jnp.cos(q[16])
	var4=jnp.sin(q[17])
	var5=0.218015
	var6=jnp.cos(q[17])
	var7=jnp.cos(q[15])
	var8=jnp.cos(q[5])
	var9=9.81
	var10=(var9*var2)
	var11=(var8*var10)
	var12=jnp.sin(q[15])
	var13=jnp.sin(q[5])
	var14=(var13*var10)
	var15=((var7*var11)-(var12*var14))
	var16=jnp.sin(q[16])
	var17=jnp.sin(q[4])
	var18=(var9*var17)
	var19=((var3*var15)-(var16*var18))
	var20=((var3*var18)+(var16*var15))
	var21=((var6*var19)-(var4*var20))
	var22=(var5*var21)
	var23=((var6*var20)+(var4*var19))
	var24=(var5*var23)
	var25=0.898919
	var26=(((var4*var22)-(var6*var24))-(var25*var20))
	var27=((var25*var19)+((var4*var24)+(var6*var22)))
	var28=0.510299
	var29=(((var3*var26)+(var16*var27))-(var28*var18))
	var30=4.801
	var31=jnp.cos(q[13])
	var32=jnp.sin(q[14])
	var33=jnp.cos(q[14])
	var34=jnp.cos(q[12])
	var35=jnp.sin(q[12])
	var36=((var34*var11)-(var35*var14))
	var37=jnp.sin(q[13])
	var38=((var31*var36)-(var37*var18))
	var39=((var31*var18)+(var37*var36))
	var40=((var33*var38)-(var32*var39))
	var41=(var5*var40)
	var42=((var33*var39)+(var32*var38))
	var43=(var5*var42)
	var44=(((var32*var41)-(var33*var43))-(var25*var39))
	var45=((var25*var38)+((var32*var43)+(var33*var41)))
	var46=(((var31*var44)+(var37*var45))-(var28*var18))
	var47=jnp.cos(q[10])
	var48=jnp.sin(q[11])
	var49=jnp.cos(q[11])
	var50=jnp.cos(q[9])
	var51=jnp.sin(q[9])
	var52=((var50*var11)-(var51*var14))
	var53=jnp.sin(q[10])
	var54=((var47*var52)-(var53*var18))
	var55=((var47*var18)+(var53*var52))
	var56=((var49*var54)-(var48*var55))
	var57=(var5*var56)
	var58=((var49*var55)+(var48*var54))
	var59=(var5*var58)
	var60=(((var48*var57)-(var49*var59))-(var25*var55))
	var61=((var25*var54)+((var48*var59)+(var49*var57)))
	var62=(((var47*var60)+(var53*var61))-(var28*var18))
	var63=jnp.cos(q[7])
	var64=jnp.sin(q[8])
	var65=jnp.cos(q[8])
	var66=jnp.cos(q[6])
	var67=jnp.sin(q[6])
	var68=((var66*var11)-(var67*var14))
	var69=jnp.sin(q[7])
	var70=((var63*var68)-(var69*var18))
	var71=((var63*var18)+(var69*var68))
	var72=((var65*var70)-(var64*var71))
	var73=(var5*var72)
	var74=((var65*var71)+(var64*var70))
	var75=(var5*var74)
	var76=(((var64*var73)-(var65*var75))-(var25*var71))
	var77=((var25*var70)+((var64*var75)+(var65*var73)))
	var78=(((var63*var76)+(var69*var77))-(var28*var18))
	var79=((((var29-(var30*var18))+var46)+var62)+var78)
	var80=((var7*var14)+(var12*var11))
	var81=(var5*var80)
	var82=((var28*var80)+((var25*var80)+var81))
	var83=((var28*var15)+((var3*var27)-(var16*var26)))
	var84=((var34*var14)+(var35*var11))
	var85=(var5*var84)
	var86=((var28*var84)+((var25*var84)+var85))
	var87=((var28*var36)+((var31*var45)-(var37*var44)))
	var88=((var50*var14)+(var51*var11))
	var89=(var5*var88)
	var90=((var28*var88)+((var25*var88)+var89))
	var91=((var28*var52)+((var47*var61)-(var53*var60)))
	var92=((var66*var14)+(var67*var11))
	var93=(var5*var92)
	var94=((var28*var92)+((var25*var92)+var93))
	var95=((var28*var68)+((var63*var77)-(var69*var76)))
	var96=(((((var30*var14)+((var7*var82)-(var12*var83)))+((var34*var86)-(var35*var87)))+((var50*var90)-(var51*var91)))+((var66*var94)-(var67*var95)))
	var97=(((((var30*var11)+((var12*var82)+(var7*var83)))+((var35*var86)+(var34*var87)))+((var51*var90)+(var50*var91)))+((var67*var94)+(var66*var95)))
	var98=((var13*var96)+(var8*var97))
	var99=((var2*var79)+(var17*var98))
	var100=jnp.sin(q[3])
	var101=((var8*var96)-(var13*var97))
	var102=-0.00276072
	var103=3.06179e-06
	var104=0.00311745
	var105=-0.029427
	var106=-0.000993282
	var107=-0.0321003
	var108=((var106*var21)-(var107*var23))
	var109=0.213
	var110=(((var104*var19)-(var105*var20))+((var108+((var109*var6)*var24))-((var109*var4)*var22)))
	var111=(((var102*var15)-(var103*var18))+var110)
	var112=0.00276072
	var113=0.000377621
	var114=-0.00311745
	var115=0.0170318
	var116=0.000993282
	var117=-0.000206526
	var118=((var116*var80)-(var117*var23))
	var119=0.0321003
	var120=0.000206526
	var121=((var119*var80)+(var120*var21))
	var122=(((var114*var80)-(var115*var20))+((var6*var118)-(var4*var121)))
	var123=0.029427
	var124=-0.0170318
	var125=(((var123*var80)+(var124*var19))+(((var6*var121)+(var4*var118))+(var109*var81)))
	var126=0.08
	var127=(((var112*var80)-(var113*var18))+((((var3*var122)-(var16*var125))-((var126*var3)*var26))-((var126*var16)*var27)))
	var128=-0.1881
	var129=((var106*var40)-(var107*var42))
	var130=(((var104*var38)-(var105*var39))+((var129+((var109*var33)*var43))-((var109*var32)*var41)))
	var131=(((var102*var36)-(var103*var18))+var130)
	var132=-0.000377621
	var133=((var116*var84)-(var117*var42))
	var134=((var119*var84)+(var120*var40))
	var135=(((var114*var84)-(var124*var39))+((var33*var133)-(var32*var134)))
	var136=(((var123*var84)+(var115*var38))+(((var33*var134)+(var32*var133))+(var109*var85)))
	var137=-0.08
	var138=(((var112*var84)-(var132*var18))+((((var31*var135)-(var37*var136))-((var137*var31)*var44))-((var137*var37)*var45)))
	var139=((var106*var56)-(var107*var58))
	var140=(((var104*var54)-(var105*var55))+((var139+((var109*var49)*var59))-((var109*var48)*var57)))
	var141=(((var112*var52)-(var103*var18))+var140)
	var142=((var116*var88)-(var117*var58))
	var143=((var119*var88)+(var120*var56))
	var144=(((var114*var88)-(var115*var55))+((var49*var142)-(var48*var143)))
	var145=(((var123*var88)+(var124*var54))+(((var49*var143)+(var48*var142))+(var109*var89)))
	var146=(((var102*var88)-(var113*var18))+((((var47*var144)-(var53*var145))-((var126*var47)*var60))-((var126*var53)*var61)))
	var147=0.1881
	var148=((var106*var72)-(var107*var74))
	var149=(((var104*var70)-(var105*var71))+((var148+((var109*var65)*var75))-((var109*var64)*var73)))
	var150=(((var112*var68)-(var103*var18))+var149)
	var151=((var116*var92)-(var117*var74))
	var152=((var119*var92)+(var120*var72))
	var153=(((var114*var92)-(var124*var71))+((var65*var151)-(var64*var152)))
	var154=(((var123*var92)+(var115*var70))+(((var65*var152)+(var64*var151))+(var109*var93)))
	var155=(((var102*var92)-(var132*var18))+((((var63*var153)-(var69*var154))-((var137*var63)*var76))-((var137*var69)*var77)))
	var156=((((((-0.0557169*var11)-(0.00051223*var18))+((((var7*var111)-(var12*var127))-((var128*var12)*var82))-((var128*var7)*var83)))+((((var34*var131)-(var35*var138))-((var128*var35)*var86))-((var128*var34)*var87)))+((((var50*var141)-(var51*var146))-((var147*var51)*var90))-((var147*var50)*var91)))+((((var66*var150)-(var67*var155))-((var147*var67)*var94))-((var147*var66)*var95)))
	var157=-0.04675
	var158=0.04675
	var159=((((((0.0557169*var14)-(-0.021231*var18))+(((((var12*var111)+(var7*var127))+(var157*var29))-((var147*var7)*var82))+((var147*var12)*var83)))+(((((var35*var131)+(var34*var138))+(var158*var46))-((var147*var34)*var86))+((var147*var35)*var87)))+(((((var51*var141)+(var50*var146))+(var157*var62))-((var128*var50)*var90))+((var128*var51)*var91)))+(((((var67*var150)+(var66*var155))+(var158*var78))-((var128*var66)*var94))+((var128*var67)*var95)))
	var160=-3.06179e-06
	var161=(((var160*var80)+(var132*var15))+((((var3*var125)+(var16*var122))+((var137*var16)*var26))-((var137*var3)*var27)))
	var162=(((var160*var84)+(var113*var36))+((((var31*var136)+(var37*var135))+((var126*var37)*var44))-((var126*var31)*var45)))
	var163=(((var160*var88)+(var132*var52))+((((var47*var145)+(var53*var144))+((var137*var53)*var60))-((var137*var47)*var61)))
	var164=(((var160*var92)+(var113*var68))+((((var63*var154)+(var69*var153))+((var126*var69)*var76))-((var126*var63)*var77)))
	var165=((((((-0.00051223*var14)+(0.021231*var11))+((var161-((var157*var12)*var82))-((var157*var7)*var83)))+((var162-((var158*var35)*var86))-((var158*var34)*var87)))+((var163-((var157*var51)*var90))-((var157*var50)*var91)))+((var164-((var158*var67)*var94))-((var158*var66)*var95)))
	return jnp.array([((var1*var99)-(var100*var101)), ((var100*var99)+(var1*var101)), ((var2*var98)-(var17*var79)), ((var2*((var13*var156)+(var8*var159)))-(var17*var165)), ((var8*var156)-(var13*var159)), var165, var164, var149, var148, var163, var140, var139, var162, var130, var129, var161, var110, var108])