'''


Filename: ValleyRidge.py (version X)

Description: 
- Predicts Major and Minor products from two transition states in reaction involving valley ridge inflections
- In order for this code to work the Gaussian frequency calculation must be run with freq=hpmodes keyword. This will ask Gaussian to print the eigenvectors to 5 significant figures.

Usage:


Algorithm Summary:
1. Import TS1 and TS2 structures
2. Align TS1 and TS2 to minimise RMSD
3. Extract TS1 and TS2 imaginary eigenvectors
4. Translate TS2 eigenvector according to the aligned TS2 geometry
5. Find u
6. Perform QRC from g+ub (i.e. TS2 structure displaced by u in the direction of imag freq vector)

@Sanha Lee
Dec 2018
'''


############################################################
# Import Modules
############################################################


import numpy as np
import argparse
import os
import math
import sys
from rdkit import Chem
import bisect
import logging
from datetime import datetime
import itertools


############################################################
# Main Code
############################################################


def main(filename1, filename2, filename3, filename4, filename5, filename6, weight_option):

    logging.basicConfig(filename='ValleyRidge_'+filename1[:-4]+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')
    
    datetime_now = datetime.now()
    formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")
    
    print ''
    print '***********************************'
    print ''
    print 'ValleyRidge.py'
    print ''
    print 'Python code to predict selectivity from two transition states'
    print '@author: Sanha Lee'
    print 'Run date: '+formatted_datetime
    print ''
    print 'Created: ValleyRidge_'+filename1[:-4]+'.log'
    print ''
    print '***********************************'
    print ''
    
    logging.info('\n')
    logging.info('***********************************\n')
    logging.info('ValleyRidge.py\n')
    logging.info('Python code to predict selectivity from two transition states')
    logging.info('@author: Sanha Lee')
    logging.info('University of Cambridge')
    logging.info('Run date: '+formatted_datetime+'\n')
    logging.info('***********************************\n')
    logging.info('Reading TS1: '+filename1)
    logging.info('Reading TS2: '+filename2)
    logging.info('Reading P1: '+filename3)
    logging.info('Reading P2: '+filename4)
    logging.info('Reading TS1 freq: '+filename5)
    logging.info('Reading TS2 freq: '+filename6+'\n')
    
    atoms1, geometries1 = ReadGeometries(filename1)
    atoms2, geometries2 = ReadGeometries(filename2)
    atoms3, geometries3 = ReadGeometries(filename3)
    atoms4, geometries4 = ReadGeometries(filename4)

    logging.info('No. of Atoms: '+str(len(atoms1))+'\n')

    equalweightslist = SetEqualWeights(atoms1)
    trueweightslist = []

    WeightsToUse = equalweightslist[:]
    TempToUse = 298.0

    
    # -- Change weights for alignment if necessary
    if weight_option == True:
        logging.info('Alignment weight option is set to atomic weights')
        for atom in atoms1:
            trueweightslist += [GetAtomNum(atom)]
        
        WeightsToUse = trueweightslist[:]
    else:
        logging.info('Alignment weight option is set equal for all atoms')


    rdkitmolTS1 = Chem.MolFromMolFile(filename1, removeHs=False, strictParsing=False)
    if not rdkitmolTS1:
        logging.error("Could not create RDKIT mol for: ",filename1)
        logging.error('Terminating Program')
        sys.exit()
    else:
        logging.info('RDKIT Mol object successfully created for: '+filename1)
        logging.info(rdkitmolTS1)
        

    rdkitmolTS2 = Chem.MolFromMolFile(filename2, removeHs=False, strictParsing=False)
    if not rdkitmolTS2:
        logging.error("Could not create mol for: ",filename2)
        logging.error('Terminating Program')
        sys.exit()
    else:
        logging.info('Mol object successfully created for: '+filename2)
        logging.info(rdkitmolTS2)


    rdkitmolP1 = Chem.MolFromMolFile(filename3, removeHs=False, strictParsing=False)
    if not rdkitmolP1:
        logging.error("Could not create mol for: "+filename3)
        logging.error('Terminating Program')
        sys.exit()
    else:
        logging.info('Mol object successfully created for: '+filename3)
        logging.info(rdkitmolP1)


    rdkitmolP2 = Chem.MolFromMolFile(filename4, removeHs=False, strictParsing=False)
    if not rdkitmolP2:
        logging.error("Could not create mol for: "+filename4)
        logging.error('Terminating Program')
        sys.exit()
    else:
        logging.info('Mol object successfully created for: '+filename4)
        logging.info(rdkitmolP2)


    # -- Identify Orthogonal Bonds

    P1bonds = ExtractBonds(rdkitmolP1, atoms1)
    P2bonds = ExtractBonds(rdkitmolP2, atoms1)
    TS1bonds = ExtractBonds(rdkitmolTS1, atoms1)
    
    logging.info('')
    logging.info('P1bonds:')
    for item in P1bonds:
        logging.info(str(item[0])+'-'+str(item[1]))
    logging.info('')
    logging.info('P2bonds:')
    for item in P2bonds:
        logging.info(str(item[0])+'-'+str(item[1]))
    
    unidenticalP1P2, unidenticalP2P1 = IdentifyChangedBonds(P1bonds, P2bonds)
    unidenticalTS1P1, unidenticalP1TS1 = IdentifyChangedBonds(TS1bonds, P1bonds)
    unidenticalTS1P2, unidenticalP2TS1 = IdentifyChangedBonds(TS1bonds, P2bonds)
    # changed_bonds1 = bonds in bondlist1 which does not exist in bondlist2
    # changed_bonds2 = bonds in bondlist2 which does not exist in bondlist1

    logging.info('')
    logging.info('unidenticalP1P2: '+str(unidenticalP1P2))
    logging.info('unidenticalP2P1: '+str(unidenticalP2P1))
    logging.info('unidenticalTS1P1: '+str(unidenticalTS1P1))
    logging.info('unidenticalTS1P2: '+str(unidenticalTS1P2))
    logging.info('unidenticalP1TS1: '+str(unidenticalP1TS1))
    logging.info('unidenticalP2TS1: '+str(unidenticalP2TS1))
    
    orthbond1list = []
    orthbond2list = []
    
    newunidenticalP1TS1 = []
    newunidenticalTS1P1 = []
    newunidenticalP2TS1 = []
    newunidenticalTS1P2 = []    
    
    # -- Remove Duplicated bonds in all lists
    
    for item in unidenticalP1TS1:
        if item in unidenticalP1P2:
            pass
        else:
            newunidenticalP1TS1 += [item]
    
    for item in unidenticalTS1P1:
        if item in unidenticalP1P2 or item in unidenticalP1TS1:
            pass
        else:
            newunidenticalTS1P1 += [item]
        
    for item in unidenticalP2TS1:
        if item in unidenticalP2P1:
            pass
        else:
            newunidenticalP2TS1 += [item]
        
    for item in unidenticalTS1P2:
        if item in unidenticalP2P1 or item in unidenticalP2TS1:
            pass
        else:
            newunidenticalTS1P2 += [item]


    # -- Rank the bond differences found

    BondrankP1 = [] 
    BondrankP2 = []

    # CaseA: When the length of B(P1|P2) >= 1 or B(P2|P1) >= 1 
    logging.info('')
    logging.info('Case when B(P1|P2) >= 1 or B(P2|P1) >= 1')
    permutationsP1P2rank, permutationsP2P1rank = RankBonds(unidenticalP1P2, unidenticalP2P1, geometries1, geometries2, geometries3, geometries4)
    BondrankP1 = BondrankP1 + permutationsP1P2rank
    BondrankP2 = BondrankP2 + permutationsP2P1rank

    # CaseB: When the length of B(P1|P2) = 0 or B(P2|P1) = 0
    # CaseC: both B(P1|P2) >= 1 or B(P2|P1) >= 1 but needs more B(|) tests
    logging.info('')
    logging.info('Replace B(P1|P2) with B(TS1|P1)')
    permutationsTS1P1P2P1, permutationsP2P1TS1P1 = RankBonds(unidenticalTS1P1, unidenticalP2P1, geometries1, geometries2, geometries3, geometries4)
    BondrankP1 = BondrankP1 + permutationsTS1P1P2P1
    BondrankP2 = BondrankP2 + permutationsP2P1TS1P1

    logging.info('')
    logging.info('Replace B(P2|P1) with B(TS1|P2)')
    permutationsP1P2TS1P2, permutationsTS1P2P1P2 = RankBonds(unidenticalP1P2, unidenticalTS1P2, geometries1, geometries2, geometries3, geometries4)
    BondrankP1 = BondrankP1 + permutationsP1P2TS1P2
    BondrankP2 = BondrankP2 + permutationsTS1P2P1P2

    logging.info('')
    logging.info('Replace B(P1|P2) with B(P1|TS1)')
    permutationsP1TS1P2P1, permutationsP2P1P1TS1 = RankBonds(unidenticalP1TS1, unidenticalP2P1, geometries1, geometries2, geometries3, geometries4)
    BondrankP1 = BondrankP1 + permutationsP1TS1P2P1
    BondrankP2 = BondrankP2 + permutationsP2P1P1TS1

    logging.info('')
    logging.info('Replace B(P2|P1) with B(P2|TS1)')
    permutationsP1P2P2TS1, permutationsP2TS1P1P2 = RankBonds(unidenticalP1P2, unidenticalP2TS1, geometries1, geometries2, geometries3, geometries4)
    BondrankP1 = BondrankP1 + permutationsP1P2P2TS1
    BondrankP2 = BondrankP2 + permutationsP2TS1P1P2


    # test whether the algorithm is using bond1 to loop or bond2 to loop

    logging.info('')
    logging.info('BondrankP1: '+str(BondrankP1))
    logging.info('BondrankP2: '+str(BondrankP2))

    testlength = 0

    if len(BondrankP1) >= len(BondrankP2):
        testlength = len(BondrankP2)
    
    if len(BondrankP2) >= len(BondrankP1):
        testlength = len(BondrankP1)
    
    testidx = 0
    
    ##############################
    ############################## Edit here
    ##############################
    
    while testidx < testlength:
        
        orthbond1list = BondrankP1[testidx]
        orthbond2list = BondrankP2[testidx]
        
        logging.info('')
        logging.info('The algorithm is current using the following bonds for the major product analysis:')
        logging.info('orthbond1list: '+str(orthbond1list))
        logging.info('orthbond1list: '+str(orthbond2list))
        
        orthbond1list = [(int(i)-1) for i in orthbond1list]
        orthbond2list = [(int(i)-1) for i in orthbond2list]
 
    
    # -- Major Product Analysis

        eigenvectors1, imag_eigenvector1, real_eigenvectors1 = ReadEigenVec(filename5)  # RealEigenVec format: [[vector1],[vector2],...]
        eigenvectors2, imag_eigenvector2, real_eigenvectors2 = ReadEigenVec(filename6)
        forceconstants1, frequencies1 = ExtractForceConsts(filename5)  # Note: This function does not return the force constant corresponding to the imaginary frequency

        imag_eigenvector1 = [float(i) for i in imag_eigenvector1]

        real_xyzeigenvectors = []
        real_floateigenvectors = []

        for eigenvector in real_eigenvectors1:
            real_floateigenvectors.append([])
            floateigenlist = [float(item) for item in eigenvector]
            real_floateigenvectors[-1] = floateigenlist

        for eigenvector in real_eigenvectors1:
            real_lineigenvectorcomp = [float(item) for item in eigenvector]
            real_xyzeigenvectorcomp = ConvertLineartoXYZ(real_lineigenvectorcomp)
            
            real_xyzeigenvectors.append([])
            real_xyzeigenvectors[-1] = real_xyzeigenvectorcomp

        

        logging.info('')
        logging.info('-- Starting Vector Analysis --')
        logging.info('')

        logging.info('coord1-1: '+str(geometries1[orthbond1list[0]]))
        logging.info('coord1-2: '+str(geometries1[orthbond1list[1]]))
        logging.info('coord2-1: '+str(geometries1[orthbond2list[0]]))
        logging.info('coord2-2: '+str(geometries1[orthbond2list[1]]))
        logging.info('TS1_bond1: '+str(GetBondLength(geometries1, orthbond1list[0], orthbond1list[1])))
        logging.info('TS1_bond2: '+str(GetBondLength(geometries1, orthbond2list[0], orthbond2list[1])))

        # Stretch the molecule along the imaginary eigenvector and find the difference with the original atom position to find the imaginary eigenvector
        lingeometries1 = ConvertXYZtoLinear(geometries1)
        lindispTS1 = list(np.array(lingeometries1) + np.array(imag_eigenvector1)) 
        dispTS1 = ConvertLineartoXYZ(lindispTS1)

        dispcomp1 = GetBondLength(dispTS1, orthbond1list[0], orthbond1list[1]) - GetBondLength(geometries1, orthbond1list[0], orthbond1list[1])
        dispcomp2 = GetBondLength(dispTS1, orthbond2list[0], orthbond2list[1]) - GetBondLength(geometries1, orthbond2list[0], orthbond2list[1])

        Redimag_eigenvector1 = [dispcomp1,dispcomp2]
    
        logging.info('')
        logging.info('Bond Imaginary Eigenvector: '+str(Redimag_eigenvector1))
    
        Redgeometry1 = [GetBondLength(geometries1, orthbond1list[0], orthbond1list[1]),GetBondLength(geometries1, orthbond2list[0], orthbond2list[1])]
        Redgeometry2 = [GetBondLength(geometries2, orthbond1list[0], orthbond1list[1]),GetBondLength(geometries2, orthbond2list[0], orthbond2list[1])]
        Redgeometry3 = [GetBondLength(geometries3, orthbond1list[0], orthbond1list[1]),GetBondLength(geometries3, orthbond2list[0], orthbond2list[1])]
        Redgeometry4 = [GetBondLength(geometries4, orthbond1list[0], orthbond1list[1]),GetBondLength(geometries4, orthbond2list[0], orthbond2list[1])]

        p1_ = list(np.array(Redgeometry3)-np.array(Redgeometry2))
        p2_ = list(np.array(Redgeometry4)-np.array(Redgeometry2))
        g_ = list(np.array(Redgeometry2)-np.array(Redgeometry1))

        logging.info('')
        logging.info('p1_: ')
        for item in ConvertLineartoXYZ(p1_):
            logging.info(str(item).strip('[]'))
        logging.info('')
        logging.info('p2_: ')
        for item in ConvertLineartoXYZ(p2_):
            logging.info(str(item).strip('[]'))
        logging.info('')
        logging.info('imag_eigenvector:')
        for item in ConvertLineartoXYZ(imag_eigenvector1):
            logging.info(str(item).strip('[]'))
        logging.info('')
        logging.info('g_:')
        for item in ConvertLineartoXYZ(g_):
            logging.info(str(item).strip('[]'))
        logging.info('')

    # -- Test whether the TS1 eigenvector needs to be inverted
        angle_phi = abs(angle(Redimag_eigenvector1,g_))
        if angle_phi > (math.pi/2):
            logging.info('phi is '+str((180/math.pi)*angle_phi)+' vector a_ will be inverted.\n')
            Redimag_eigenvector1 = InvertVec(Redimag_eigenvector1)
            angle_phi = abs(angle(Redimag_eigenvector1,g_))
            logging.info('new phi is '+str((180/math.pi)*abs(angle(Redimag_eigenvector1,g_))))
        else:
            logging.info('phi is '+str((180/math.pi)*angle_phi)+' vector a_ will not be inverted.\n')

        angle_p1p2 = abs(angle(p1_,p2_))

        mu1_ = Findmu_(Redimag_eigenvector1, p1_, g_)
        mu2_ = Findmu_(Redimag_eigenvector1, p2_, g_)

        lambda1_ = Findlambda_(Redimag_eigenvector1, p1_, g_)
        lambda2_ = Findlambda_(Redimag_eigenvector1, p2_, g_)

        if np.sign(mu1_) != np.sign(mu2_) and lambda1_ > 0 and lambda2_ > 0:
            logging.info('+mu_, -mu_, +lambda_, +lambda_ combination found. The algorithm wil continue.')
            break
        else:
            logging.info('')
            logging.info('The sign test did not find +mu_, -mu_, +lambda_, +lambda_ combination. The algorithm will try different set of bonds')
            testidx += 1

    logging.info('')
    logging.info('angle(p1,p1): '+str((180/math.pi)*angle_p1p2))
    logging.info('|p1|: '+str(length(p1_)))
    logging.info('|p2|: '+str(length(p2_)))
    logging.info('|a|: '+str(length(imag_eigenvector1)))
    logging.info('|b|: '+str(length(imag_eigenvector2)))
    logging.info('|g|: '+str(length(g_)))
    logging.info('')
    logging.info('phi:')
    logging.info(angle(imag_eigenvector1,g_)*(180/math.pi))


    # -- Result Analysis --

    if np.sign(mu1_) != np.sign(mu2_) and lambda1_ > 0 and lambda2_ > 0:
 
        # -- Real Eigenvector Proudct Ratio Analysis

        BondRealEigenvectors = ProjectEigenToBond(real_floateigenvectors, orthbond1list, orthbond2list, geometries1)
        logging.info('')
        logging.info('BondRealEigenVectors')
        for item in BondRealEigenvectors:
            logging.info(item)

        BondRealEigenvectorLengths = [] # Find the length of the real eigenvectors projected to the two key bonds
        
        for vector in BondRealEigenvectors:
            BondRealEigenvectorLengths += [length(vector)]

        TempToUse = 298.0
        HalfWellWidths = CalcXs(forceconstants1, TempToUse, frequencies1)

        logging.info('')        
        logging.info('HalfWellWidths: ')
        for item in HalfWellWidths:
            logging.info(item)

        logging.info('') 
        logging.info('BondRealEigenvectorLengths:')
        for item in BondRealEigenvectorLengths:
            logging.info(item)

        constantAs = []

        for index in range(len(BondRealEigenvectors)):
            const_A = HalfWellWidths[index]/BondRealEigenvectorLengths[index]
            constantAs += [const_A]

        logging.info('')             
        logging.info('const_As: ')
        for item in constantAs:
            logging.info(item)

        m_list = ImagOrthogonalProj(BondRealEigenvectors,constantAs,Redimag_eigenvector1,g_)

        ConstBs = FindConstantB(m_list,Redimag_eigenvector1,g_)

        logging.info('')         
        logging.info('ConstBs')
        for item in ConstBs:
            logging.info(item)
        
        major_length, minor_length = FindProdRatio(m_list, ConstBs, Redimag_eigenvector1)

        major_ratio = major_length/(major_length + minor_length)
        minor_ratio = minor_length/(major_length + minor_length)
        
        if length(g_) < 0.5 and angle_phi > 20 and angle_phi < 50:
            logging.info('')         
            logging.info('current major ratio:'+str(major_ratio))
            logging.info('current minor ratio:'+str(minor_ratio))
            logging.info('|g| < 0.5 and 20 < phi < 50 degs, the algorithm will change the major:minor product ratio')
            major_ratio = (major_ratio - 0.58369)/0.4029
            
            if major_ratio > 1:
                major_ratio = 1
            
            minor_ratio = 1 - major_ratio
        
        
        '''
        if length(g_) < 0.5 and ((180/math.pi)*angle_phi) < 20:
            logging.info('')         
            logging.info('current major ratio:'+str(major_ratio))
            logging.info('current minor ratio:'+str(minor_ratio))
            logging.info('|g| < 0.5 and phi < 20 degs, the algorithm will change the major:minor product ratio')
            major_ratio = (major_ratio - 0.1543659)/0.7437851
            minor_ratio = 1 - major_ratio
            
        elif length(g_) < 0.5 and 20 < ((180/math.pi)*angle_phi) and ((180/math.pi)*angle_phi) < 40:
            logging.info('')         
            logging.info('|g| < 0.5 and 20 < phi < 40 degs, the algorithm will change the major:minor product ratio')
            major_ratio = (major_ratio - 0.5417960)/0.4733665
            minor_ratio = 1 - major_ratio
        '''
        
        if mu1_ > 0 and mu2_ < 0:
            logging.info('')
            logging.info('**** Analysis Completed ****')
            logging.info('Major product is '+str(filename3))
            logging.info('Minor product is '+str(filename4))
            print ''
            print '**** Analysis Completed ****'
            print 'Major product is '+str(filename3)
            print 'Minor product is '+str(filename4)
        elif mu2_ > 0 and mu1_ < 0:
            logging.info('')
            logging.info('**** Analysis Completed ****')
            logging.info('Major product is '+str(filename4))
            logging.info('Minor product is '+str(filename3))
            print ''
            print '**** Analysis Completed ****'
            print 'Major product is '+str(filename4)
            print 'Minor product is '+str(filename3)

        logging.info('')
        logging.info('mu1_ = '+str(mu1_))
        logging.info('mu2_ = '+str(mu2_))
        logging.info('lambda1_ = '+str(lambda1_))
        logging.info('lambda2_ = '+str(lambda2_))
        logging.info('|g_| = '+str(length(g_)))
        if length(g_) > 1.0:
            logging.info('WARNING: |g_| is large (|g_| > 1)')       
        logging.info('phi = '+str((180/math.pi)*angle_phi))
        logging.info('')
        logging.info('The algorith will now proceed to estimate the major and minor product ratios')
        logging.info('')
        logging.info('Product Ratio Calculation Completed:')
        logging.info('Major Product : Minor Product ratio')
        logging.info(str(round(major_ratio*100, 1))+' : '+str(round(minor_ratio*100, 1))+'\n')
        if round(major_ratio*100, 1) < 60.0:
            logging.info('WARNING: the predicted ratio is close to 50:50')
        logging.info('****************************')
        logging.info('')
        
        print ''
        print 'mu1_ = '+str(mu1_)
        print 'mu2_ = '+str(mu2_)
        print 'lambda1_ = '+str(lambda1_)
        print 'lambda2_ = '+str(lambda2_)
        print '|g_| = '+str(length(g_))
        if length(g_) > 1.0:
            print 'WARNING: |g_| is large (|g_| > 1)'
        print 'phi = '+str((180/math.pi)*angle_phi)
        print ''
        print 'Product Ratio Calculation Completed:'
        print 'Major Product : Minor Product ratio'
        print str(round(major_ratio*100, 1))+' : '+str(round(minor_ratio*100, 1))
        if round(major_ratio*100, 1) < 60.0:
            print 'WARNING: the predicted ratio is close to 50:50'
        print ''
        print '****************************'
        print ''
     

    else:
            logging.error('')
            logging.error('**** Analysis Completed ****')
            logging.error('+mu_, -mu_, +lambda_, +lambda_ combination not found')
            logging.error('ERROR: Unable to predict the major product')
            logging.error('')
            logging.error('mu1_ = '+str(mu1_))
            logging.error('mu2_ = '+str(mu2_))
            logging.error('lambda1_ = '+str(lambda1_))
            logging.error('lambda2_ = '+str(lambda2_))
            logging.info('|g_| = '+str(length(g_)))
            logging.info('phi = '+str((180/math.pi)*angle_phi))
            logging.error('')
            logging.error('The algorith will not proceed to estimate the major and minor product ratios')
            logging.error('****************************')

            print ''
            print '**** Analysis Completed ****'
            print '+mu_, -mu_, +lambda_, +lambda_ combination not found'
            print 'ERROR: Unable to predict the major product'
            print ''
            print 'mu1_ = '+str(mu1_)
            print 'mu2_ = '+str(mu2_)
            print 'lambda1_ = '+str(lambda1_)
            print 'lambda2_ = '+str(lambda2_)
            print '|g_| = '+str(length(g_))
            print 'phi = '+str((180/math.pi)*angle_phi)
            print ''
            print 'The algorith will not proceed to estimate the major and minor product ratios'
            print '****************************'
            print ''
            
            

############################################################
# Functions
############################################################


def SetEqualWeights(atom_list):
    '''
    Sets weightings used for geometry alignments
    '''
    weights = []
    for item in atom_list:
        weights += [1]
    
    return weights


def InvertVec(vector):
    InvertedVector = [(-1)*i for i in vector]
    return InvertedVector


def ConvertLineartoXYZ(LinearList):
    '''
    Convert [x1,y1,z1,x2,y2,z2,...] format to [[x1,y1,z1],[x2,y2,z2],...]
    '''
    XYZlist = []
    i = 0

    while i < len(LinearList):
        XYZlist.append(LinearList[i:i+3])
        i+=3

    return XYZlist


def VectorAddition(vec1,vec2):

    addedvector = []

    for i in range(len(vec1)):
        addition = float(vec1[i]) + float(vec2[i])
        addedvector += [addition]

    return addedvector


def ConvertXYZtoLinear(XYZlist):
    '''
    Covert geometry list in [[x1,y1,z1],[x2,y2,z2],...] format to [x1,y1,z1,x2,y2,z2,...] format
    '''
    linearlist = []

    for i in range(len(XYZlist)):
        for j in range(3):
           linearlist.append(XYZlist[i][j])

    return linearlist


def Findmu_(vec_a, vec_b, vec_g):
    '''
    Finds u in point of closest approach analysis
    '''
    nominator = dotproduct(vec_a,vec_g)*dotproduct(vec_a,vec_b) - dotproduct(vec_b,vec_g)*dotproduct(vec_a,vec_a)
    denominator = dotproduct(vec_a,vec_a)*dotproduct(vec_b,vec_b)-(dotproduct(vec_a,vec_b)**2)

    return (nominator/denominator)


def Findlambda_(vec_a, vec_b, vec_g):
    '''
    Finds lambda in point of closest approach analysis
    '''
    nominator = dotproduct(vec_a,vec_g)*dotproduct(vec_b,vec_b) - dotproduct(vec_b,vec_g)*dotproduct(vec_a,vec_b)
    denominator = dotproduct(vec_a,vec_a)*dotproduct(vec_b,vec_b)-(dotproduct(vec_a,vec_b)**2)

    return (nominator/denominator)


def dotproduct(v1, v2):
    '''
    Returns dot product of vectors v1 and v2
    '''
    return sum((float(a)*float(b)) for a, b in zip(v1, v2))


def length(v):
    '''
    Returns magnitude of vector v
    '''
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    '''
    Returns angle between vectors v1 and v2
    '''
    return math.acos(round(dotproduct(v1, v2) / (length(v1) * length(v2)),5))


def FindUnitVector(vector):
    '''
    Returns the unit vector
    
    '''
    unit_vector = []
    magnitude = length(vector)
    
    for item in vector:
        unit_vector += [item/magnitude]
    
    return unit_vector
    

def ReadGeometries(GMolFile):
    '''
    Finds optimisation steps in the Gaussian output file and extracts the coordinates for each step.
    If the output file is a frequency calculation only one geometry exists.
    Returns the coordinates as a list
    '''
    gausfile = open(GMolFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()
    
    index = 0
    atoms = []
    coords = []
      
    for index in range(len(GOutp)):
        if index > 3:
            if len(GOutp[index].split()) < 8:
                break
            else:
                data = GOutp[index].split()
                atoms.append(data[3])
                coords.append([float(x) for x in data[0:3]])
    
    return atoms, coords


def FindDispVec(Geom1,Geom2):
    '''
    The atoms must be written in same order in Geom1 and Geom2. Geom1 and Geom2 must be same molecular species otherwise this function will not work.
    '''
    DispVec = []

    for coord in range(len(Geom1)):
        coorddisp = Geom2[coord] - Geom1[coord]
        DispVec.append(coorddisp)

    return DispVec


def ReadEigenVec(GOutpFile): 
    '''
    ReadEigenVec: returns eigenvectors in ascending order (as printed in Gaussian output file) of frequency. 
    The format is [[Eigenvector1],[Eigenvector2],...]. Edit the return at the end of the function if the 
    structure is not at the transition state.
    '''
    gausfile = open(GOutpFile, 'r')
    GOutp = gausfile.readlines()
    gausfile.close()

    index = 0
    EigenIndexes = []
    EigenVectors = []

    for index in range(len(GOutp)):
        if "Coord Atom Element:" in GOutp[index]:
            EigenIndexes.append(index + 1)

    for i, gindex in enumerate(EigenIndexes):

        first_line = True # not ideal location for this line

        for line in GOutp[gindex:]:
            if '                         ' in line: # find
                first_line = True
                break
            if 'Harmonic frequencies (cm**-1)' in line: # check for final few eigenvectors
                break
            else:
                data = line.split()

                if first_line == True:
                    eigenvec_element = [[] for i in range(len(data)-3)]
                    first_line = False

                for index in range(len(data[3:])):
                    eigenvec_element[index].append(data[index+3])

        for item in eigenvec_element:
            EigenVectors += [item]

    imag_eigenvector = EigenVectors[0]
    real_eigenvectors = EigenVectors[1:]

    return EigenVectors, imag_eigenvector, real_eigenvectors


Lookup = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
          'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', \
          'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', \
          'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', \
          'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', \
          'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']


def GetAtomSymbol(AtomNum):
    '''
    Extracts the element symbol from Lookup list
    '''
    if AtomNum > 0 and AtomNum < len(Lookup):
        return Lookup[AtomNum-1]
    else:
        logging.info( "No such element with atomic number " + str(AtomNum))
        return 0


def GetAtomNum(AtomSym):
    '''
    Extracts the atomic number 
    '''
    if AtomSym in Lookup:
        return Lookup.index(AtomSym)+1
    else:
        logging.info( "No such element with atomic symbol " + AtomSym)
        return 0


#calculates the RMS between the two molecules
def RMSMol(mol1, mol2, w):
    e = [0.0, 0.0, 0.0]
    
    for i in range(0, len(mol1)):
        e[0] += w[i] * (mol1[i][0] - mol2[i][0])**2
        e[1] += w[i] * (mol1[i][1] - mol2[i][1])**2
        e[2] += w[i] * (mol1[i][2] - mol2[i][2])**2
    
    return math.sqrt(sum(e)/sum(w))


def ExtractForceConsts(GOutpFile):
    '''
    Extracts the real force constants and the real frequencies from the frequency calculation
    '''
    gausfile = open(GOutpFile, 'r')
    GOutpFile = gausfile.readlines()
    gausfile.close()

    index = 0
    ForceConsts = []
    ForceConstsIndexes = []
    Frequencies = []
    FrequencyIndexes = []

    for index in range(len(GOutpFile)):
        if "Force constants ---" in GOutpFile[index]:
            ForceConstsIndexes.append(index)
            FrequencyIndexes.append(index-2)

    for index, line in enumerate(GOutpFile):
        
        if index in FrequencyIndexes:
            data = line.split()
            FreqOnlyData = data[2:]
            
            for freq in FreqOnlyData:
                Frequencies.append(float(freq))
        
        if index in ForceConstsIndexes:
            data = line.split()
            FConlydata = data[3:]
            
            for fc in FConlydata:
                ForceConsts.append(float(fc))

    return ForceConsts[1:], Frequencies[1:]  # do not return the first force constant which corresponds to the imaginary normal mode


def QuantisedE(ClassicalE, Frequency):
    '''
    Returns quantised vibrational energy
    '''
    EnergyList = []
    PlankConst = 6.62607e-34
    AvogadroConst = 6.02214e23
    n_quantum = 0.0
    
    while n_quantum < 500:
        EnergyList += [PlankConst*(Frequency*100*299792458*AvogadroConst)*(0.5+n_quantum)]
        n_quantum += 1

    QuantumE = 0.0

    if ClassicalE > EnergyList[0]:
        listindex = bisect.bisect(EnergyList, ClassicalE)
        QuantumE = EnergyList[listindex-1]
    elif ClassicalE <= EnergyList[0]:
        QuantumE = EnergyList[0]
    
    return QuantumE


def GetBondLength(geometry, atomidx1, atomidx2):
    
    atomcoord1 = np.array(geometry[atomidx1])
    atomcoord2 = np.array(geometry[atomidx2])
    bondvector = atomcoord2 - atomcoord1
    bondlength = length(bondvector)
    
    #print 'atomidx1 ', atomidx1
    #print 'atomidx2 ', atomidx2
    #print 'atomcoord1 ', atomcoord1
    #print 'atomcoord2 ', atomcoord2
    #print 'bondlength ', bondlength
    
    return bondlength


def ExtractBonds(rdkitmol, atomlist):
    '''
    Extract all bonds in the rdkit molecule and the atom indexes of atoms in the bond
    
    '''
    bondlist = []
    
    for bond in rdkitmol.GetBonds():
        bondlist.append([])
        bondlist[-1] = [bond.GetBeginAtomIdx()+1, bond.GetEndAtomIdx()+1]

    return bondlist


def IdentifyChangedBonds(bondlist1, bondlist2):
    '''
    Find the uncommon bonds in the two product bond lists
    
    '''
    changed_bonds1 = []
    changed_bonds2 = []
    
    for bond_1 in bondlist1:
        bond_1.sort()
        
    for bond_2 in bondlist2:
        bond_2.sort()
    
    for bond_1 in bondlist1:
        if bond_1 in bondlist2:
            pass
        else:
            changed_bonds1 += [bond_1]
    
    for bond_2 in bondlist2:
        if bond_2 in bondlist1:
            pass
        else:
            changed_bonds2 += [bond_2]
    
    return changed_bonds1, changed_bonds2  
    # changed_bonds1 = bonds in bondlist1 which does not exist in bondlist2
    # changed_bonds2 = bonds in bondlist2 which does not exist in bondlist1


def TestBondProjSign(vector1, vector2, bondvector):
    
    vector_sign = 1
    component_to_return = 0
    projdiff = list(np.array(vector2) - np.array(vector1))
    
    logging.info('---')
    logging.info('length(vector1) '+str(length(vector1)))
    logging.info('length(vector2) '+str(length(vector2)))
    logging.info('vector1 '+str(vector1))
    logging.info('vector2 '+str(vector2))
    logging.info('dotproduct(vector1,vector2): '+str(dotproduct(vector1, vector2)))
    logging.info('angle(vector2,vector1) '+str(round(angle(vector2, vector1), 2)))
    logging.info('angle(vector2,bondvector) '+str(round(angle(vector2, bondvector), 2)))
    logging.info('angle(projdiff,bondvector) '+str(round(angle(projdiff, bondvector), 2)))
    
    if round(angle(vector2, vector1), 2) == round(math.pi, 2) :
        
        if round(angle(vector2, bondvector), 2) == round(math.pi, 2):
            vector_sign*-1
            
        elif round(angle(vector2, bondvector), 2) == 0:
            pass
            
        else:
            logging.error('')
            logging.error('****')
            logging.error('ERROR: The angle between the projected vector2 and the bond vector are not 0 or 180 degrees')
            logging.error('The program will terminate')
            logging.error('****')
            logging.error('')
            sys.exit()
            
        component_to_return = length(vector1) + length(vector2)
        component_to_return = component_to_return * vector_sign
        
    elif round(angle(vector2, vector1), 2) == 0.00 :
        
        projdiff = list(np.array(vector2) - np.array(vector1))
        
        if length(vector2) > length(vector1):
            
            if round(angle(projdiff, bondvector), 2) == 0.00:
                pass
            elif round(angle(projdiff, bondvector), 2) == round(math.pi, 2):
                vector_sign*-1
            else:
                logging.error('')
                logging.error('****')
                logging.error('ERROR: The angle between the difference in the projected vectors and the bond vector are not 0 or 180 degrees')
                logging.error('The program will terminate')
                logging.error('****')
                logging.error('')
                sys.exit()
                
        elif length(vector1) > length(vector2):

            if round(angle(projdiff, bondvector), 2) == round(math.pi, 2):
                vector_sign*-1
            elif round(angle(projdiff, bondvector), 2) == 0.00:
                pass
            else:
                logging.error('')
                logging.error('****')
                logging.error('ERROR: The angle between the difference in the projected vectors and the bond vector are not 0 or 180 degrees')
                logging.error('The program will terminate')
                logging.error('****')
                logging.error('')
                sys.exit()                
            
        else:
            logging.error('')
            logging.error('****')
            logging.error('ERROR: The angle between the projected vector2 and the bond vector are not 0 or 180 degrees')
            logging.error('The program will terminate')
            logging.error('****')
            logging.error('')
            sys.exit()
        
        component_to_return = length(vector1) + length(vector2)
        component_to_return = component_to_return * vector_sign
        
    else:
        logging.error('')
        logging.error('****')
        logging.error('ERROR: The angle between the projected vector1 and vector2 are not 0 or 180 degrees')
        logging.error('The program will terminate')
        logging.error('****')
        logging.error('')
        sys.exit()
    
    return component_to_return


def ProjectEigenToBond(real_eigenlist, orthbond1, orthbond2, TS1geometry):
    '''
    Function to project the real eigenvector component to the two orthogonal bonds.
    Note the orthbond1 and orthbond2 has 1 subtracted from the atom indices
    
    '''

    bondeigenvectors = []  # [[bond1, bond2], ...] etc for each real eigenvectors
    lingeometries1 = ConvertXYZtoLinear(TS1geometry)
    
    for eigenvector in real_eigenlist: # loops through each real eigenvector list of list
        

        lindispTS1 = list(np.array(lingeometries1) + np.array(eigenvector)) 
        dispTS1 = ConvertLineartoXYZ(lindispTS1)

        dispcomp1 = GetBondLength(dispTS1, orthbond1[0], orthbond1[1]) - GetBondLength(TS1geometry, orthbond1[0], orthbond1[1])
        dispcomp2 = GetBondLength(dispTS1, orthbond2[0], orthbond2[1]) - GetBondLength(TS1geometry, orthbond2[0], orthbond2[1])

        Redreal_eigenvector1 = [dispcomp1,dispcomp2]
    
        bondeigenvectors.append([])
        bondeigenvectors[-1] = Redreal_eigenvector1
        
    return bondeigenvectors



def CalcXs(FClist, Temp, FrequencyList):
    '''
    Calculate half valley widths required for product ratio predictions
    '''
    Xs = []
    ClassicalEnergy = 1.5*8.3144598*Temp  # (3/2)*RT
    BohrtoAngstrom = 0.529177 #Angstroms
    
    # Note:
    # Gaussian force constants are given in mdyne/A
    # 1 Hart/Bohr^-2 = 15.569141 mdyne/A
    # 1 Hart = 2625.5002 kJ/mol
    # 1 Bohr = 0.529177 Angstrom
    
    
    FClist_J = []
    
    for item in FClist:
        FClist_J += [item*(1/15.569141)*2625500.2*(1/BohrtoAngstrom)**2]  # units J mol^-1 A^-2
    
    for index, item in enumerate(FClist_J):
        freq_i = FrequencyList[index]
        X = np.sqrt((QuantisedE(ClassicalEnergy,freq_i)*2)/(item)) # QuantisedE in J/mol,
        Xs.append(X)
    
    return Xs
    

def ImagOrthogonalProj(real_bondeigenlist, constA_list, imaginary_eigenvector, vector_g):
    
    orthimageigen1 = [-1*imaginary_eigenvector[1], imaginary_eigenvector[0]]
    orthimageigen2 = [imaginary_eigenvector[1], imaginary_eigenvector[0]*-1]
    unit_orthimageigen1 = FindUnitVector(orthimageigen1)
    unit_orthimageigen2 = FindUnitVector(orthimageigen2)

    m1 = []
    m2 = []
    m_return = []
    m1_lengths = []
    
    for index in range(len(real_bondeigenlist)):
        
        m1_eigen = list( (dotproduct((constA_list[index]*np.array(real_bondeigenlist[index])), orthimageigen1) / length(orthimageigen1)) * np.array(unit_orthimageigen1))
        m2_eigen = list( (dotproduct((constA_list[index]*np.array(real_bondeigenlist[index])), orthimageigen2) / length(orthimageigen2)) * np.array(unit_orthimageigen2) *-1)
        
        m1.append([])
        m2.append([])
        m_return.append([])
        
        m1[-1] = m1_eigen
        m2[-1] = m2_eigen
        m_return[-1] = [m1_eigen, m2_eigen]
        
    
    for item in m1:
        m1_lengths += [length(item)]
    
    logging.info('')
    logging.info('m1_lengths:')
    for item in m1_lengths:
        logging.info(item)
    
    return m_return


def FindConstantB(m_vectorlist, imaginary_eigenvector, vec_g): # m_vectorlist is in format [[[bond1m1,bond1m2],[bond2m1,bond2m2]], ... etc]
    
    constantBs = []
    
    for m_vector in m_vectorlist:
        
        constB = dotproduct(m_vector[0], vec_g) / dotproduct(m_vector[0], m_vector[0])
        
        if constB > 0:
            constantBs += [constB]
        
        if constB < 0:
            constB = dotproduct(m_vector[1], vec_g) / dotproduct(m_vector[1], m_vector[1])
            constantBs += [constB]
    
    return constantBs


def FindProdRatio(m_vectorlist, constB_list, imaginary_eigenvector):
    
    majP_total = 0
    minP_total = 0
    
    logging.info('')
    
    for index in range(len(constB_list)):
        
        if constB_list[index] < 1:
            majP = length(m_vectorlist[index][0]) + length( list( np.array(m_vectorlist[index][0]) * constB_list[index] ))
            minP = length(m_vectorlist[index][0]) - length( list( np.array(m_vectorlist[index][0]) * constB_list[index] ))
            logging.info('majP: '+str(majP))
            logging.info('minP: '+str(minP))
            majP_total += majP
            minP_total += minP
        if constB_list[index] > 1:
            majP = length(m_vectorlist[index][0])*2
            logging.info('majP: '+str(majP))
            logging.info('minP: '+'0')
            majP_total += majP
    
    return majP_total, minP_total


def RankBonds(bondlist1, bondlist2, TS1geom, TS2geom, P1geom, P2geom):
    '''
    Function finds all permutations of bondlistP1 and bondlistP2 and ranks the bonds by finding the angles
    theta1 and theta2. (theta1 = angle between -g_ vector and p1 vector, theta2 = angle between -g_ vector and p2 vector)
    '''
    
    # 'bondlist' is the list of bonds to sort
    # 'bondchangelist' is the list of magnitude of change in bondlength between currentgeom and comparegeom
    # 'sortedbondchange' is the bondchangelist organised in descending order
    # 'bondchangeidx' is the list of index of bondchangelist in the same order as sortedbondchange

    rankedbond1 = []
    rankedbond2 = []

    if len(bondlist1) == 0 or len(bondlist2) == 0:
        pass
    else:
    
        bondpermutations = list(itertools.product(bondlist1,bondlist2))
        bondpermutations = [list(item) for item in bondpermutations]
        bondanglestype = []
        bondangles = []
        permutationsremove = []
    
        for bondpair in bondpermutations:
            testbond1 = bondpair[0]
            testbond2 = bondpair[1]  
            if testbond1 == testbond2:
                permutationsremove += [bondpair]
    
        for item in permutationsremove:
            bondpermutations.remove(item)
    
        for bondpair in bondpermutations:
        
            testbond1 = bondpair[0]
            testbond2 = bondpair[1]        
            testbond1 = [i - 1 for i in testbond1]
            testbond2 = [i - 1 for i in testbond2]
            RedTS1geom = [GetBondLength(TS1geom, testbond1[0], testbond1[1]),GetBondLength(TS1geom, testbond2[0], testbond2[1])]
            RedTS2geom = [GetBondLength(TS2geom, testbond1[0], testbond1[1]),GetBondLength(TS2geom, testbond2[0], testbond2[1])]
            RedP1geom = [GetBondLength(P1geom, testbond1[0], testbond1[1]),GetBondLength(P1geom, testbond2[0], testbond2[1])]
            RedP2geom = [GetBondLength(P2geom, testbond1[0], testbond1[1]),GetBondLength(P2geom, testbond2[0], testbond2[1])]

            p1_ = list(np.array(RedP1geom)-np.array(RedTS2geom))
            p2_ = list(np.array(RedP2geom)-np.array(RedTS2geom))
            neg_g_ = list(np.array(RedTS1geom)-np.array(RedTS2geom))

            theta1 = angle(neg_g_, p1_)
            theta2 = angle(neg_g_, p2_)
            
            bondangle1 = abs((180/math.pi)*theta1 - 90)
            bondangle2 = abs((180/math.pi)*theta2 - 90)
            bondanglecomb = bondangle1 + bondangle2
            
            if (180/math.pi)*theta1 > 90 and (180/math.pi)*theta2 > 90:
                bondanglestype += ['o']
            else: 
                bondanglestype += ['a']
            bondangles += [bondanglecomb]

        sortedbondangles = sorted(bondangles)
        bondangleidx = []
        rankedbonds = []
        rankedtypes = []
        rankedacute = []
        rankedobtuse = []

        for item in sortedbondangles:
            bondangleidx += [bondangles.index(item)]
    
        for item in bondangleidx:
            rankedbonds += [bondpermutations[item]]
            rankedtypes += [bondanglestype[item]]
        
        for index in range(len(rankedbonds)):
            if rankedtypes[index] == 'o':
                rankedobtuse += [rankedbonds[index]]
            if rankedtypes[index] == 'a':
                rankedacute += [rankedbonds[index]]
        
        rankedbonds = rankedobtuse + rankedacute

        for item in rankedbonds:
            rankedbond1 += [item[0]]
            rankedbond2 += [item[1]]

        logging.info('bondpermutations: '+str(bondpermutations))

    logging.info('rankedbond1 '+str(rankedbond1))
    logging.info('rankedbond2 '+str(rankedbond2))

    return rankedbond1, rankedbond2


############################################################
# Execution
############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for analyzing geometry optimisations from a Gaussian output file') # create parser object
    parser.add_argument('GausFile1', help="Gaussian mol output file No.1: TS1") 
    parser.add_argument('GausFile2', help="Gaussian mol output file No.2: TS2")
    parser.add_argument('GausFile3', help="Gaussian mol output file No.3: P1")
    parser.add_argument('GausFile4', help="Gaussian mol output file No.4: P2")
    parser.add_argument('GausFile5', help="Gaussian frequency output file for TS1")
    parser.add_argument('GausFile6', help="Gaussian frequency output file for TS2")
    parser.add_argument("-w", "--weight_option", help='use atomic weight rather than setting all atom weights equal when performing molecule alignement', action='store_true')

    args = parser.parse_args() # inspects command line, convert each argument to the appropriate type and invoke the appropriate action


    if not os.path.isfile(args.GausFile1):
        print args.GausFile + " is not a valid file."
        quit()

    if not os.path.isfile(args.GausFile2):
        print args.GausFile + " is not a valid file."
        quit()

    if not os.path.isfile(args.GausFile3):
        print args.GausFile + " is not a valid file."
        quit()

    if not os.path.isfile(args.GausFile4):
        print args.GausFile + " is not a valid file."
        quit()
        
    if not os.path.isfile(args.GausFile5):
        print args.GausFile + " is not a valid file."
        quit()
    
    if not os.path.isfile(args.GausFile6):
        print args.GausFile + " is not a valid file."
        quit()

    main(args.GausFile1, args.GausFile2, args.GausFile3, args.GausFile4, args.GausFile5, args.GausFile6, args.weight_option)



