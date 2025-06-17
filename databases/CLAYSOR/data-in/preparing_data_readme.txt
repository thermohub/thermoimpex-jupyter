Export from GEMS to JSON
	- IComp relevant for STDB - sites and elements
	- DComp relevant for STDB - all aqeuous and sorption species
- export elements all
	- in text editor add coresponding DComp  master  "IC_DC": "Clai-", "EssOH"
	- add class x and s for exchange and surface special elements
		
all in one file
- edit in text editor the key of the data 
	- in text replace "U" with "X" to surface species - rdc code
	- in text replace "M" with "EX" to exchange species - rdc code


#- export dcomp aq, illite, mont
#- export reacdc aq, illiteX, illiteSCM, montX, montSCM


Am+3
Ba+2
Ca+2
Cd+2
Cm+3
Co+2
Cs+2
Eu+3
Fe+2
H+
K+
Mg+2
Mn+2
Na+
Nb(OH)4+
NH4+
Ni+2
Np+4
NpO2+
PaO(OH)+2
Pb+2
Pu+3
Pu+4
Ra+2
Sn+4
Sr+2
Th+4
U+4
UO2+
UO2+2
Zn+2


After export - by text replacement do

iSs to Ilt_s
iSv to Ilt_v
iSw to Ilt_w

Ss to Mnt_s
Sv to Mnt_v
Sw to Mnt_w

space before number

(?<![A-Za-z])(\d)([A-Z])

$1 $2

