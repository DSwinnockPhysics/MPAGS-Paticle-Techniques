from __future__ import division
import numpy
import uproot
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import itertools
import scipy.optimize

def yielder(x):
    if isinstance(x,tuple):
        for y in x:
            if isinstance(y,tuple):
                yield from yielder(y)
            else:
                yield y
    else:
        yield x
     

def GaussianForFit(x, *p):
    Amp, Mean, StandardDeviation = p
    return Amp*numpy.exp(-(x-Mean)**2/(2.0*StandardDeviation**2))

def CalculateSquareDifference(ObservedData,ExpectedData): 
    return numpy.sum((ObservedData-ExpectedData)**2)
     

def FindOptimalBestFitLine(InputArrayX,InputArrayY,CombinationsToTest):
    BestSquareDifference = 0 # Initialise
    BestCoeffs = (0,0)
    index = 0
    for i in CombinationsToTest:
        InputArraySubsetX,InputArraySubsetY = [],[]
        for j in i:
            # First make a subset of 5 of the hits to make a best fit line for
            InputArraySubsetX.append(InputArrayX[j])
            InputArraySubsetY.append(InputArrayY[j])
        InputArraySubsetX = numpy.array(InputArraySubsetX)
        InputArraySubsetY = numpy.array(InputArraySubsetY)
        # Then find the coefficients for this line
        CurrentCoeffs = numpy.polyfit(InputArraySubsetX,InputArraySubsetY,deg=1)
        BestFitLineYs = InputArraySubsetX*CurrentCoeffs[0] + CurrentCoeffs[1] # y=mx+c
        SquareDifference = CalculateSquareDifference(InputArraySubsetY,BestFitLineYs)
        if index == 0: # For the first one just replace (0,0) with the calculated Coefficients
            BestSquareDifference = SquareDifference
            BestCoeffs = CurrentCoeffs
            index+=1
        else:
            if SquareDifference<BestSquareDifference:
                BestSquareDifference=SquareDifference
                BestCoeffs = CurrentCoeffs                    
    # After they've all been tested, return the coefficients of the best line
    return BestCoeffs
                    
           

def FindBestFitLine(InputArrayX,InputArrayY):
    if len(InputArrayX)==5:
        return numpy.polyfit(InputArrayX,InputArrayY,deg=1)
    else:
        # If there are multiple hits, find the line of best fit that best matches 5 of the points
        ElementsSeen = []
        ElementsToCheck = []
        NonRepeatedElements = []
        for i in InputArrayX: # If the number has already appeared, add it to the numbers to check as long as it's not already in there
            if i in ElementsSeen:
                if i not in ElementsToCheck:
                    ElementsToCheck.append(i)
            else:
                ElementsSeen.append(i)
        for i in range(len(InputArrayX)): # Make a list containing the indices of non-repeated values
            if InputArrayX[i] not in ElementsToCheck:
                NonRepeatedElements.append(i)
        NumberOfRepeatedNumbers = len(ElementsToCheck)
        IndicesOfRepeats = []
        for i in range(NumberOfRepeatedNumbers): # For each repeated number make a list of the indices where it appears
             IndicesOfRepeats.append(numpy.where(InputArrayX == ElementsToCheck[i])[0].tolist())
        CombinationsToAdd = IndicesOfRepeats[0]
        if NumberOfRepeatedNumbers > 1:
            for i in range(NumberOfRepeatedNumbers-1):
                CombinationsToAdd = list(itertools.product(CombinationsToAdd,IndicesOfRepeats[i+1]))
        CombinationsForFit = []
        for i in CombinationsToAdd:
            CurrentList = []
            YieldedValues = yielder(i)
            for k in YieldedValues:
                CurrentList.append(k)
            CombinationsForFit.append(CurrentList) 
        for i in CombinationsForFit:
            i+=NonRepeatedElements
        BestCoeffs = FindOptimalBestFitLine(InputArrayX, InputArrayY, CombinationsForFit)
        return BestCoeffs
    

def FindLineIntersection(Coeffs1,Coeffs2):
    ZIntersect = (Coeffs2[1]-Coeffs1[1])/(Coeffs1[0]-Coeffs2[0]) # Z_int = (c2-c1)/(m1-m2)
    XIntersect = Coeffs1[0]*ZIntersect + Coeffs1[1] # x = mz+c
    return numpy.array([ZIntersect,XIntersect])


def FindBFIntersection(Coeffs1,Coeffs2):
    FirstIntersect = numpy.array([-1,Coeffs1[0]*-1+Coeffs1[1]]) # Calculate line of best fit at z=-1
    SecondIntersect = numpy.array([1,Coeffs2[0]*1+Coeffs2[1]]) # Calculate second line at z=1
    MissedBField = False # Initialise
    
    if SecondIntersect[1] > 1: # If the line is above the field at z=1, check the top of the BField
        SecondIntersect = numpy.array([(1-Coeffs2[1])/Coeffs2[0],1]) # Check the top line
        if (SecondIntersect[0] < -1): # If the line doesn't intersect the B Field
            MissedBField = True
    
    if SecondIntersect[1] < -1: # If the line is below the field at z=1, check the bottom of the BField
        SecondIntersect = numpy.array([(-1-Coeffs2[1])/Coeffs2[0],-1]) # Check the bottom line
        if SecondIntersect[0] < -1: # If the line doesn't intersect the B Field
            MissedBField = True
    
    if (FirstIntersect[1] > 1 or FirstIntersect[1] < -1): # If the particle misses the B Field at z=-1
        MissedBField = True
    return FirstIntersect, SecondIntersect, MissedBField


def FindPerpendicular(FitCoeffs,FieldIntercept):
    PerpCoeffs = numpy.array([0,0]) # [m,c]
    PerpCoeffs[0] = -1./FitCoeffs[0] # -1/m
    # c = y+x/m
    PerpCoeffs[1] = FieldIntercept[1] + FieldIntercept[0]/FitCoeffs[0]
    return PerpCoeffs


B = 0.5 # Magnetic Field/Tesla
q = 1.602*10**-19 # Electron charge
c = 299792458 # speed of light, m/s
L = 2 # Length of B Field in metres

file = uproot.open(r"C:\Users\nemes\Desktop\B5_Custom.root")

print(file["B5"])
Dc1HitsX = file["B5"]["Dc1HitsVector_x"].array()/1000
Dc1HitsZ = file["B5"]["Dc1HitsVector_z"].array()/2-6
Dc2HitsX = file["B5"]["Dc2HitsVector_x"].array()/1000
Dc2HitsZ = file["B5"]["Dc2HitsVector_z"].array()/2+2.5
ECEnergies = file["B5"]["ECEnergy"].array()
HCEnergies = file["B5"]["HCEnergy"].array()
print(file["B5"].keys())
                  

nevents = 1000

FitCoeffsDc1 = numpy.zeros((nevents,2))
FitCoeffsDc2 = numpy.zeros((nevents,2))
LineIntersections = numpy.zeros((nevents,2))
FieldIntersections1 = numpy.zeros((nevents,2))
FieldIntersections2 = numpy.zeros((nevents,2))
PerpCoeffs1 = numpy.zeros((nevents,2))
PerpCoeffs2 = numpy.zeros((nevents,2))
CircleCentres = numpy.zeros((nevents,2))
CircleRadii = numpy.zeros((nevents,1))
Momenta = numpy.zeros((nevents,1))
MomentaGeV = numpy.zeros((nevents,1))

nEventsSkipped = 0
nEventsMore5Hits = 0

for i in range(nevents):
    
    # First fit the line of best fit
    FitCoeffsDc1[i] = FindBestFitLine(numpy.array(Dc1HitsZ[i]),numpy.array(Dc1HitsX[i])) # vector of coefficients (gradient, intercept)
    FitCoeffsDc2[i] = FindBestFitLine(numpy.array(Dc2HitsZ[i]),numpy.array(Dc2HitsX[i]))    
    
    LineIntersections[i] = FindLineIntersection(FitCoeffsDc1[i], FitCoeffsDc2[i])
        
    FieldIntersections1[i], FieldIntersections2[i], MissedBField = FindBFIntersection(FitCoeffsDc1[i],FitCoeffsDc2[i]) # Get the points at which the lines of best fit enter the B field
    if MissedBField==True:
        nEventsSkipped+=1
        continue # If the particle misses the B field, skip the event since no momentum information can be gained
    
    # PERPENDICULAR METHOD ---------------------------------------------------
    # Calculate Perpendicular lines
    PerpCoeffs1[i] = FindPerpendicular(FitCoeffsDc1[i],FieldIntersections1[i])
    PerpCoeffs2[i] = FindPerpendicular(FitCoeffsDc2[i],FieldIntersections2[i])
    
    # Find their intercept
    CircleCentres[i] = FindLineIntersection(PerpCoeffs1[i], PerpCoeffs2[i])
    CircleRadii[i] = numpy.linalg.norm(CircleCentres[i]-FieldIntersections1[i])
    # ------------------------------------------------------------------------
    
    # ANGLE METHOD
    DeltaX = 1-LineIntersections[i,0] # Edge of BField - Intersection x position
    DeltaY = numpy.abs(FieldIntersections2[i,1] - (FitCoeffsDc1[i,0]*1+FitCoeffsDc1[i,1]))
    Theta = numpy.abs(numpy.arctan(DeltaY/DeltaX))
    Momenta[i] = B*q*L/(2*numpy.sin(Theta/2))
    MomentaGeV[i] = B*0.3*2/2/numpy.sin(Theta/2)
    
    
    
    
# Get momenta
MomentaPerp = CircleRadii*B*q # Kgm/s
MomentaGeVPerp = MomentaPerp*c/(q*10**9)

    
print("Skipped %s events due to missing field." % nEventsSkipped)  
indextoplot = 998
# print("FieldIntersect2:",FieldIntersections2[indextoplot])
# print("Fit2:",FitCoeffsDc2[indextoplot])
# print("Perp2:",PerpCoeffs2[indextoplot])
# print("Radius:",CircleRadii[indextoplot])
# print("Momentum:",Momenta[indextoplot]) # kgm/s
# print("Momentum(Perp):",MomentaPerp[indextoplot]) # kgm/s
# print("Momentum GeV:",MomentaGeV[indextoplot]) # GeV
# print("Momentum GeV(Perp):",MomentaGeVPerp[indextoplot]) # GeV
# print("#Momenta=",numpy.shape(MomentaGeV))
# print("Circle Centre:",CircleCentres[indextoplot])


# Plot tracks
ExLineZs = numpy.linspace(-6,2)
ExLineXs = FitCoeffsDc1[indextoplot,0]*ExLineZs + FitCoeffsDc1[indextoplot,1]
Ex2LineZs = numpy.linspace(6,-2)
Ex2LineXs = FitCoeffsDc2[indextoplot,0]*Ex2LineZs + FitCoeffsDc2[indextoplot,1]
# PerpLine1Zs = numpy.linspace(-4,4)
# PerpLine2Zs = numpy.linspace(-4,4)
# PerpLine1Xs = PerpCoeffs1[indextoplot,0]*PerpLine1Zs + PerpCoeffs1[indextoplot,1]
# PerpLine2Xs = PerpCoeffs2[indextoplot,0]*PerpLine2Zs + PerpCoeffs2[indextoplot,1]
rectangle = patches.Rectangle((-1,-1),2,2,linewidth=1,edgecolor='r',facecolor='none')
fig,ax = pyplot.subplots(nrows=1, ncols=1, figsize=[9,3])
ax.plot(ExLineZs,ExLineXs,linewidth=1,color='b')
ax.plot(Ex2LineZs,Ex2LineXs,linewidth=1,color='orange')
#ax.plot(PerpLine1Zs,PerpLine1Xs,linewidth=1,color='g')
#ax.plot(PerpLine2Zs,PerpLine2Xs,linewidth=1,color='g')
ax.add_patch(rectangle)
ax.set_ylim(-2,2)
ax.set_xlim(-6,6)
ax.scatter(Dc1HitsZ[indextoplot],Dc1HitsX[indextoplot])
ax.scatter(Dc2HitsZ[indextoplot],Dc2HitsX[indextoplot])
ax.scatter(LineIntersections[indextoplot,0],LineIntersections[indextoplot,1],color='r')
ax.scatter(FieldIntersections1[indextoplot,0],FieldIntersections1[indextoplot,1],color='b')
ax.scatter(FieldIntersections2[indextoplot,0],FieldIntersections2[indextoplot,1],color='orange')
#ax.scatter(CircleCentres[indextoplot,0],CircleCentres[indextoplot,1],color='g')
pyplot.show()
pyplot.close(fig)



MomentaHist,BinEdges = numpy.histogram(MomentaGeV,bins=10,range=(97,104))
# print(MomentaHist,BinEdges)
BinWidth = (BinEdges[1]-BinEdges[0])
BinCentres = (BinEdges+BinWidth/2)
BinCentres = BinCentres[:-1]
# print(MomentaHist,BinCentres)
MomentaHist = numpy.array(MomentaHist)
BinCentres = numpy.array(BinCentres)
ResultsOfFit, var_matrix = scipy.optimize.curve_fit(GaussianForFit,BinCentres,MomentaHist,p0=[510,140,16],maxfev=1000)
print(ResultsOfFit) # [Amplitude, Mean, Standard Deviation]
FitXs = numpy.linspace(90,110,num=1000)
FitYs = GaussianForFit(FitXs,*ResultsOfFit)
pyplot.plot(FitXs,FitYs,color='b')

MomentaHistPerp,BinEdgesPerp = numpy.histogram(MomentaGeVPerp,bins=9,range=(92,109))
#
BinWidthPerp = BinEdgesPerp[1]-BinEdgesPerp[0]
BinCentresPerp = (BinEdgesPerp+BinWidthPerp/2)
BinCentresPerp = BinCentresPerp[:-1]
#
MomentaHistPerp = numpy.array(MomentaHistPerp)
BinCentresPerp = numpy.array(BinCentresPerp)
ResultsOfFitPerp, var_matrix_Perp = scipy.optimize.curve_fit(GaussianForFit,BinCentresPerp,MomentaHistPerp,p0=[450,100,2],maxfev=1000)
print(ResultsOfFitPerp)
FitXsPerp = numpy.linspace(90,110,num=1000)
FitYsPerp = GaussianForFit(FitXsPerp,*ResultsOfFitPerp)
pyplot.plot(FitXsPerp,FitYsPerp,color='r')

pyplot.hist(MomentaGeVPerp,bins=BinEdgesPerp,color='r',label="Direct r",histtype='step')
pyplot.hist(MomentaGeV,bins=BinEdges,label="Angle->r",histtype='step')
pyplot.legend()

MomentumResolution = ResultsOfFit[2]/ResultsOfFit[1]
MomentumResolutionPerp = ResultsOfFitPerp[2]/ResultsOfFitPerp[1]

print("Momentum resolution (angle):",MomentumResolution,"GeV")
print("Momentum resolution (perp):",MomentumResolutionPerp,"GeV")
