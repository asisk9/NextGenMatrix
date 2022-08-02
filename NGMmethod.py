
"""
Code for calculating the basic reproductive number using the Next Generation
Matrix method for models 1-3 in the manuscript "A Network Theorectic Method 
for the Basic Reproductive Number for Infectious Diseases".

Created: Aug. 2022
Authors: Anna Sisk (asisk9@vols.utk.edu), Nina Fefferman
If you use this code please cite: Anna Sisk and Nina Fefferman "A Network 
Theorectic Method for the Basic Reproductive Number for Infectious Diseases"

******************Functions******************

eq_points: finds equilibrium points
Arguments: the LHS of the model, the state variables

NextGen: finds the basic reproductive number using the next generation method
Arguments: State variable for the infected class(es), state variables, terms 
that represent inflow into the infected classes, other terms in the infected 
class(es), disease free equilibrium

Also includes a main script where two models are defined, the functions are 
used, and the results are printed
"""

import sympy as sym
from sympy.solvers.solveset import nonlinsolve

#Function definitions
def eq_points(system,state_var):
    eq_pts=nonlinsolve(system, state_var)

    return eq_pts

def NextGen(InfectVar,state_var,InFlow,OFlow,DFE):
    f=sym.Matrix(InFlow)
    v=sym.Matrix(OFlow)
   
    F=f.jacobian(InfectVar)
    V=v.jacobian(InfectVar)
    
    for i in range(len(InfectVar)):
        F=F.subs(state_var[i],DFE[i])

    for i in range(len(InfectVar)):
        V=V.subs(state_var[i],DFE[i])  
    InverseV=sym.Inverse(V)
    NextGen=F*InverseV
    return NextGen.eigenvals()

######################
# Main Script
######################

#pick which model to run by changing the i value
i=1

if i==1:
#SIR model 1
    #State variables
    S = sym.Symbol('S')
    I = sym.Symbol('I')
    R = sym.Symbol('R')
    state_var=[S, I, R]
    
    #Parameters
    b = sym.Symbol('b')
    g = sym.Symbol('g')
    
    #Model definition
    Sdot = -b*S*I
    Idot = b*S*I-g*I
    Rdot = g*I
    system=[Sdot, Idot, Rdot]
    
if i==2:
#SEIR- model 2
    #State variables
    S = sym.Symbol('S')
    I = sym.Symbol('I')
    E = sym.Symbol('E')
    R = sym.Symbol('R')
    state_var=[S, I, E, R]
    
    #Parameters
    b = sym.Symbol('b')
    g = sym.Symbol('g')
    k = sym.Symbol('k')
    
    #Model definition
    Sdot = -b*S*I
    Edot = b*S*I-k*E
    Idot = k*E-g*I
    Rdot = g*I
    system=[Sdot, Idot, Edot, Rdot]
    
if i==3:
# SEIR with vital dynamics- model 3
    #State variables
    S = sym.Symbol('S')
    I = sym.Symbol('I')
    E = sym.Symbol('E')
    R = sym.Symbol('R')
    state_var=[S, I, E, R]
    
    #Parameters
    b = sym.Symbol('b')
    g = sym.Symbol('g')
    k = sym.Symbol('k')
    u = sym.Symbol('u')
    N = sym.Symbol('N')
    
    #Model definition
    Sdot = u*N-u*S-(b*S*I)/N
    Edot = (b*S*I)/N-(k+u)*E
    Idot = k*E-(g+u)*I
    Rdot = g*I-u*R
    system=[Sdot, Idot, Edot, Rdot]  

######################
# calling functions
# ######################

eq=list(eq_points(system,state_var))

if i==1:
    basic_repro=NextGen([I],state_var,[b*S*I],[g*I],eq[0])
if i==2:
    basic_repro=NextGen([E, I],state_var,[b*S*I, 0],[k*E, g*I-k*E],eq[0])
if i==3:
    basic_repro=NextGen([E, I],state_var,[(b*S*I)/N,0],[(u+k)*E,
                                                        (g+u)*I-k*E],eq[0])


######################## 
# printing 
# ######################
sym.pprint('The disease free equilibrium point is {}.'.format(eq[0]))
print('The basic reproductive number is '  + str(next(iter((basic_repro))))+
'.')

