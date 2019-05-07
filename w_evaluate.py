#!/usr/bin/python3

#=================================  HEADER  ====================================

import way_cnn.s2_validation

#-------------------------------------------------------------------------------




#==================================  MAIN  =====================================

acc1 = []
for i in range(10,40,5):
	acc.append( way_cnn.s2_validation.exec("mel",i,200) )
	print( acc1 )

acc2 = []
for i in range(50,200,25):
	acc.append( way_cnn.s2_validation.exec("mel",25,i) )
	print( acc2 )

print("Testes:")
print("\t",acc1)
print("\t",acc2)

#-------------------------------------------------------------------------------
