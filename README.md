# EXOPLANET DATA ANALYSIS AND MODELLING PROJECT

predicting model & visualization for exoplanets prediction based on dataset for exoplanets. 
dataset: [http://exoplanet.eu/catalog/all_fields/#](url)


**HYPOTHESIS // EXPECTATIONS**
- Mass: 0.1 to 317 earth masses
- Radius: less than 1 to 11.2 earth radii
- Orbital period: 1 to +365 days
- Star distance: 0.01 to 1000 parsecs
- Star mass: 0.1 to 5 sun masses
- Star age: 0.01 to 10 Gy


**DATA CONCLUSIONSSS for exoplanets from dataset**
- Mass: avg) 0.1 to 1.7 x earth mass
- Radius: avg) 0.05 to 1.6 x earth radius
- Orbital period: avg) 0.3 to 42 days
- Star distance: avg) 14 to 1300 parsecs
- Star mass: avg) 0.01 to 1.75 x sun mass
- Star age: avg) 0.04 to 11 Gy (giga [billion] years) —> main sequence phase


**Correlations (from heat-map)**
- (0.7) star_teff & star_metallicity
- (0.49) star_mass & star_teff
- (0.43) star_age & star_metallicity

- (-0.56) mag_v & mag_k


**Problems**
  Number of iterations reached limit, for pairplot

**Other**
orbital period, radius, mass, and host star properties & planet, planet status, planet det 
<Period(day) , Radius (RJup/Rearth), <Mass(MJup/MEarth
&    	Star name :
     <   α2000 (hh :mm :ss) : Right Ascension
     <   δ2000 (hh :mm :ss) : Declination
  *    < mV : apparent magnitude in the V band
    *  < mI : apparent magnitude in the I band
     * <  mJ : apparent magnitude in the J band
    *  <  mH : apparent magnitude in the H band
     * <  mK : apparent magnitude in the K band
 ***      < Distance (pc) : distance of the star to the observer
***       < Metallicity : decimal logarithm of the massive elements (« metals ») to hydrogen ratio in solar units  (i.e. <Log [(metals/H)star/(metals/H)Sun] )
****     <   Mass (Msun) : star mass in solar units
***      <  Radius (Rsun) : star radius in solar units
   ***    < Sp. Type : stellar spectral type
   ***    < Age (Gy) : stellar age
  ***     < Teff : effective stellar temperature 
    *   < Detected disc :  (direct imaging or IR excess) disc detected
 ***      < Magnetic field (Yes/No) : stellar magnetic field detected








# **MODEL RESULTS conclusion**
_Best;_ RandomForestClassifier even with n_estimators =100, then KNclassifier & MLP
_Worst;_ SVC





GaussianNB 
GaussianProcessClassifier 
RandomForestClassifier (best)
LogisticRegression
SVC (worst)
KNeighborsClassifier (2 best)
MLPClassifier (2 best)





# MODEL RESULTS

GaussianNB
Matrix1: 
 [[ 444   75]
 [  45 1059]]
Accuracy1:  92.60628465804066

GaussianProcessClassifier
Matrix2: 
 [[480  39]
 [346 758]]
Accuracy2:  76.2784966112138

RandomForestClassifier (best)
Matrix3: 
 [[ 470   49]
 [  14 1090]]
Accuracy3:  96.11829944547135

LogisticRegression
Matrix4: 
 [[ 448   71]
 [  55 1049]]
Accuracy4:  92.2365988909427

SVC (worst)
Matrix5: 
 [[   0  519]
 [   1 1103]]
Accuracy5:  67.96056685150955

KNeighborsClassifier (2 best)
Matrix6: 
 [[ 457   62]
 [  59 1045]]
Accuracy6:  92.54467036352435

MLPClassifier (2 best)
Matrix7: 
 [[474  45]
 [151 953]]
Accuracy7:  87.92359827479976
