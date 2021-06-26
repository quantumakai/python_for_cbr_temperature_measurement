import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import qmfsb_constants as cons

fname = 'datafiles\Cobe.csv'
c0 = (cons.wdr * cons.k) / cons.h
Ti = 1.
icnt = 0
imax = 1000
eps = 1.
thresh = 1.e-9
beta = 1.0

def get_cobe_data(fname):
    ndat = []
    lam = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        lcnt = 0
        for row in csv_reader:
            if (lcnt > 0) :
                lam.append(float(row[0]))
                ndat.append(float(row[1]))
            lcnt += 1
    return np.array(lam), np.array(ndat), lcnt

fmeasv0, measv, mcnt = get_cobe_data(fname)
lamv = (1. / fmeasv0) * 1.e-2
fmeasv = cons.c / lamv
dmax = max(measv)

def P(f, T):
    coef = (8. * math.pi * cons.h) / (cons.c **3)
    num = f**3
    den = math.exp((cons.h * f) / (cons.k * T)) - 1.
    return coef * (num/den)

def Pprime(f, T):
    u = (cons.h * f) / (cons.k * T)
    coef = (8. * math.pi * cons.h**2) / (cons.k * cons.c **3)
    fT1 = 1./ T**2
    num = (f**4) * math.exp(u)
    den = (math.exp(u) - 1.) ** 2
    return coef * fT1 * (num / den)

def Pprime2(f, T):
    u = (cons.h * f) / (cons.k * T)
    coef = (8. * math.pi * cons.h ** 2 * f**4) / (cons.k * cons.c **3)
    fT1 = -2. / T**3
    num1 = math.exp(u)
    den1 = (math.exp(u) - 1.)**2
    f1 = coef * fT1 * (num1 / den1)
    fT2 = 1. / T**2
    num20 = den1 * num1 * u * (-1. / T)
    num21 = 2. * num1 * (math.exp(u) - 1.) * num1 * u * (-1. / T)
    num2 = num20 - num21
    den2 = den1 * den1
    f2 = coef * fT2 * (num2 / den2)
    return f1 + f2

# Temp dependent maximum functions
def Pmax(T):
    coeff = (8. * math.pi * cons.h * (c0**3)) / ((cons.c**3) * (math.exp(cons.h * c0 / cons.k) - 1 ))
    return coeff * T**3

def Pmaxprime(T):
    coeff = (8. * math.pi * cons.h * (c0**3)) / ((cons.c**3) * (math.exp(cons.h * c0 / cons.k) - 1 ))
    return 3. * coeff * T * T

def PmaxPrime2(T):
    coeff = (8. * math.pi * cons.h * (c0**3)) / ((cons.c**3) * (math.exp(cons.h * c0 / cons.k) - 1 ))
    return 6. * coeff * T

# Normalized Plancks Distribution and derivatives
def Pnorm(f, T):
    return P(f, T) / Pmax(T)

def Pnormprime(f, T):
    f1 = Pprime(f, T) / Pmax(T)
    f2 = -(P(f, T) / (Pmax(T) * Pmax(T))) * Pmaxprime(T)
    return f1 + f2

def Pnormprime2(f, T):
    f1 = (-1. / (Pmax(T) * Pmax(T))) * Pprime(f, T) * Pmaxprime(T)
    f2 = (1. / Pmax(T)) * Pprime2(f, T)
    f3 = Pprime(f, T) * ( -1. / (Pmax(T) * Pmax(T))) * Pmaxprime(T)
    f4 = P(f, T) * (2. / (Pmax(T)**3) * (Pmaxprime(T)**2))
    f5 = -P(f, T) * (1. / (Pmax(T) * Pmax(T))) * PmaxPrime2(T)
    return f1 + f2 + f3 + f4 + f5

def Newtonfun(fv, dv, T):
    mysum = 0.
    idx = 0
    for (f, d) in zip(fv, dv):
        mysum = mysum + 2. * (Pnorm(f, T) - d) * Pnormprime(f, T)
        idx += 1
    return mysum

def Newtonfunprime(fv, dv, T):
    mysum = 0.
    for (f, d) in zip(fv, dv):
        mysum = mysum + 2.*Pnormprime(f, T)**2 + 2.*(Pnorm(f, T) - d) * Pnormprime(f, T)
    return mysum

# Setup reconstruction
Tmin = 1.e-1
Tmax = 10.
noT = 100
NewtonFunv = []
Tv = np.linspace(Tmin, Tmax, noT)
for T in Tv:
    NewtonFunv.append(Newtonfun(fmeasv, measv/dmax, T))

# Compute solution using Newton's method
Tip1v = [Ti]
errv = []
delT = []

while eps > thresh and icnt < imax:
    print('Error: ', eps, ' Threshold: ', thresh, 'Temp: ', Ti)
    Tip1 = Ti - beta * (Newtonfun(fmeasv, measv/dmax, Ti) / Newtonfunprime(fmeasv, measv/dmax, Ti) )
    Tip1v.append(Tip1)
    delT.append(Newtonfun(fmeasv, measv/dmax, Ti) / Newtonfunprime(fmeasv, measv/dmax, Ti))
    eps = abs((Tip1 - Ti) / Tip1 )
    errv.append(eps)
    Ti = Tip1
    icnt += 1

yv = []
fv = []

for (f, d) in zip(fmeasv, measv/dmax):
    yv.append(Pnorm(f, Ti))
    fv.append(f)

if (icnt < imax):
    print("We have converged in ", icnt, "iterations.")
    print("Estimated root/temperature: ", Ti)
else:
    print("Warning : Max iterations reached before convergence.")
    print("Last temperature value: ", Ti)

# Figures to plot

figure, axis = plt.subplots(2, 2)
axis[0, 0].plot(fv, yv, 'k+:', label='fit')
axis[0, 0].plot(fmeasv, measv/dmax, 'ro', label= 'exp')
axis[0, 0].set_title("Data and Fit(line)")
axis[0, 0].set(xlabel="frequency Hz", ylabel = "P/Pmax")
axis[0, 0].legend()

axis[0, 1].plot(Tip1v, 'bo-.')
axis[0, 1].plot(0.*np.array(Tip1v) + Ti, 'k--')
axis[0, 1].set_title("Temp Converge: " + "%.3f" % Ti + " K")
axis[0, 1].set(xlabel= "Iterations", ylabel= "Temp K")

axis[1, 0].plot(errv, 'r--')
axis[1, 0].plot(0.*np.array(Tip1v) + thresh, 'g--')
axis[1, 0].text(.4 * icnt , 1.4 * thresh, 'threshold')
axis[1, 0].set(xlabel= "Iternations", ylabel= "Error")

axis[1, 1].plot(Tv, np.array(NewtonFunv), 'r')
axis[1, 1].plot(Tv, 0.*np.array(NewtonFunv), 'k--')
axis[1, 1].set(xlabel= "Temp K", ylabel= "f(T)")
plt.show()


