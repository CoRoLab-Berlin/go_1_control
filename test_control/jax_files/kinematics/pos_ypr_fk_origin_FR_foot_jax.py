import jax.numpy as jnp
import numpy as np
import jax
@jax.jit
def pos_ypr_fk_origin_FR_foot(x):
	var1=1
	var2=2
	var3=jnp.cos((x[3]/var2))
	var4=jnp.sin((x[4]/var2))
	var5=(var3*var4)
	var6=jnp.cos((x[5]/var2))
	var7=jnp.sin((x[3]/var2))
	var8=jnp.cos((x[4]/var2))
	var9=(var7*var8)
	var10=jnp.sin((x[5]/var2))
	var11=((var5*var6)+(var9*var10))
	var12=((var9*var6)-(var5*var10))
	var13=-0.0935
	var14=(var3*var8)
	var15=(var7*var4)
	var16=((var14*var10)-(var15*var6))
	var17=((var14*var6)+(var15*var10))
	var18=-0.16
	var19=jnp.cos((x[6]/var2))
	var20=jnp.sin((x[6]/var2))
	var21=((var11*var19)+(var12*var20))
	var22=((var17*var20)+(var16*var19))
	var23=((var17*var19)-(var16*var20))
	var24=((var12*var19)-(var11*var20))
	var25=-0.426
	var26=jnp.cos((x[7]/var2))
	var27=jnp.sin((x[7]/var2))
	var28=((var23*var26)-(var21*var27))
	var29=((var23*var27)+(var21*var26))
	var30=((var22*var27)+(var24*var26))
	var31=((var22*var26)-(var24*var27))
	var32=jnp.cos((x[8]/var2))
	var33=jnp.sin((x[8]/var2))
	var34=((var28*var32)-(var29*var33))
	var35=((var28*var33)+(var29*var32))
	var36=((var31*var33)+(var30*var32))
	var37=((var31*var32)-(var30*var33))
	var38=0.3762
	var39=-0.213
	return jnp.array([((((x[0]+((0.1881*((var1-(var2*jnp.square(var11)))-(var2*jnp.square(var12))))+(var13*((var11*var16)-(var17*var12)))))+(var18*((var21*var22)-(var23*var24))))+(var25*((var28*var29)+(var30*var31))))+(var25*((var34*var35)+(var36*var37)))), ((((x[1]+((var38*((var17*var12)+(var11*var16)))+(-0.04675*((var1-(var2*jnp.square(var16)))-(var2*jnp.square(var12))))))+(-0.08*((var1-(var2*jnp.square(var22)))-(var2*jnp.square(var24)))))+(var25*((var30*var29)-(var28*var31))))+(var25*((var36*var35)-(var34*var37)))), ((((x[2]+((var38*((var12*var16)-(var17*var11)))+(var13*((var17*var16)+(var12*var11)))))+(var18*((var23*var22)+(var24*var21))))+(var39*((var1-(var2*jnp.square(var31)))-(var2*jnp.square(var29)))))+(var39*((var1-(var2*jnp.square(var37)))-(var2*jnp.square(var35))))), ajnp.tan2((var2*((var34*var36)+(var37*var35))),(var1-(var2*(jnp.square(var35)+jnp.square(var36))))), ajnp.sin((var2*((var34*var35)-(var36*var37)))), ajnp.tan2((var2*((var34*var37)+(var35*var36))),(var1-(var2*(jnp.square(var37)+jnp.square(var35)))))])