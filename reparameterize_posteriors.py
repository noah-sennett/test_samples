"""Convert a posterior distribution of dchi in the parameterization used with SEOBNRT to a posterior distribution of dchi in the parameterization used with PhenomPNRT"""

import numpy as np
import lal
import argparse
import re
from scipy import random
from scipy.special import lambertw
from scipy import interpolate

#Component mass range for runs in BNS TGR paper
compmin = 0.5
compmax = 7.73105475907

#Dimensionless spin range for runs in BNS TGR paper
spinmin = -0.99
spinmax = 0.99

#The functions phi${N} return the coefficient of the N/2-PN term in the inspiral (as in Eq. A4 of https://arxiv.org/abs/1005.3306)

def phi0(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return 1.

def phi1(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return 1.

def phi2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  eta = (m1*m2)/(m1+m2)**2.
  return 5.*(743./84. + 11.*eta)/9.

def phi3(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  return -16.* np.pi + 188.*SL/3. + 25.*dSigmaL

def phi4(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  #qm_def are the spin susceptibailities of the objects, which we take as the black hole value of 1. These enter in the "quadrupole-monopole" terms.
  qm_def1 = 1
  qm_def2 = 1
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  pnsigma = eta * (721./48. * a1L * a2L - 247./48. * a1dota2) + (720.*(qm_def1) - 1.)/96.0* m1M* m1M * a1L * a1L + (720. *(qm_def2) - 1.)/96.0 * m2M * m2M * a2L * a2L - (240.*(qm_def1) - 7.)/96.0 * m1M * m1M * a1sq - (240.*(qm_def2) - 7.)/96.0 * m2M * m2M * a2sq
  
  return 5.*(3058.673/7.056 + 5429./7.*eta + 617.*eta*eta)/72. - 10.*pnsigma

def phi5l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  pngamma = (554345./1134. + 110.*eta/9.)*SL + (13915./84. - 10.*eta/3.)*dSigmaL
  return 5./3. * (7729./84. - 13. * eta) * np.pi - 3. * pngamma

def phi6(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  #qm_def are the spin susceptibailities of the objects, which we take as the black hole value of 1. These enter in the "quadrupole-monopole" terms.
  qm_def1 = 1
  qm_def2 = 1
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  pnss3 = (326.75/1.12 + 557.5/1.8*eta) * eta * a1L * a2L + ((4703.5/8.4 + 2935./6. * m1M - 120. * m1M * m1M)*(qm_def1) + (-4108.25/6.72 - 108.5/1.2*m1M + 125.5/3.6*m1M*m1M))*m1M*m1M* a1sq + ((4703.5/8.4 + 2935./6. * m2M - 120. * m2M * m2M)*(qm_def2) + (-4108.25/6.72 - 108.5/1.2*m2M + 125.5/3.6*m2M*m2M))*m2M*m2M* a2sq
  return (11583.231236531/4.694215680 - 640./3. * np.pi * np.pi - 6848./21.*lal.GAMMA) + eta*(-15737.765635/3.048192 + 2255./12.*np.pi*np.pi) + eta*eta*76055./1728. - eta*eta*eta*127825./1296. + (-6848./21.)*np.log(4.) + np.pi*(3760.*SL + 1490*dSigmaL)/3. + pnss3

def phi6l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return -6848./21.

def phi7(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  m1M = m1/(m1+m2)
  m2M = m2/(m1+m2)
  d = (m1-m2)/(m1+m2)
  eta = (m1*m2)/(m1+m2)**2.
  SL = m1M * m1M * a1L + m2M * m2M * a2L
  dSigmaL = d * (m2M * a2L - m1M * a1L)
  return np.pi*(77096675./254016. + 378515./1512.*eta - 74045./756.*eta*eta) + (-8980424995./762048. + 6586595.*eta/756. - 305.*eta*eta/36.)* SL - (170978035./48384. - 2876425.*eta/672. - 4735.*eta*eta/144.)* dSigmaL

def phiMinus2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return 1.

#The functions phi${N}NS return the spin-independent component of the coefficient of the N/2-PN term in the inspiral

def phi0NS(m1,m2):
  return phi0(m1,m2,0.,0.,0.,0.,0.)

def phi1NS(m1,m2):
  return phi1(m1,m2,0.,0.,0.,0.,0.)

def phi2NS(m1,m2):
  return phi2(m1,m2,0.,0.,0.,0.,0.)

def phi3NS(m1,m2):
  return phi3(m1,m2,0.,0.,0.,0.,0.)

def phi4NS(m1,m2):
  return phi4(m1,m2,0.,0.,0.,0.,0.)

def phi5lNS(m1,m2):
  return phi5l(m1,m2,0.,0.,0.,0.,0.)

def phi6NS(m1,m2):
  return phi6(m1,m2,0.,0.,0.,0.,0.)

def phi6lNS(m1,m2):
  return phi6l(m1,m2,0.,0.,0.,0.,0.)

def phi7NS(m1,m2):
  return phi7(m1,m2,0.,0.,0.,0.,0.)

def phiMinus2NS(m1,m2):
  return phiMinus2(m1,m2,0.,0.,0.,0.,0.)

#The functions phi${N}S return the spin-dependent component of the coefficient of the N/2-PN term in the inspiral

def phi0S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi0(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi0NS(m1, m2)

def phi1S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi1(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi1NS(m1, m2)

def phi2S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi2NS(m1, m2)

def phi3S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi3(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi3NS(m1, m2)

def phi4S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi4(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi4NS(m1, m2)

def phi5lS(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi5l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi5lNS(m1, m2)

def phi6S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi6(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi6NS(m1, m2)

def phi6lS(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi6l(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi6lNS(m1, m2)

def phi7S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phi7(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phi7NS(m1, m2)

def phiMinus2S(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2):
  return phiMinus2(m1, m2, a1L, a2L, a1sq, a2sq, a1dota2) - phiMinus2NS(m1, m2)


#Dictionaries that map the testing-GR parameter of each run to the corresponding function above
phiDict = {'dchi0':phi0, 'dchi1':phi1, 'dchi2':phi2, 'dchi3':phi3, 'dchi4':phi4, 'dchi5l':phi5l, 'dchi6':phi6, 'dchi6l':phi6l, 'dchi7':phi7, 'dchiminus2':phiMinus2, 'dipolecoeff':phiMinus2}
phiNSDict = {'dchi0':phi0NS, 'dchi1':phi1NS, 'dchi2':phi2NS, 'dchi3':phi3NS, 'dchi4':phi4NS, 'dchi5l':phi5lNS, 'dchi6':phi6NS, 'dchi6l':phi6lNS, 'dchi7':phi7NS, 'dchiminus2':phiMinus2NS, 'dipolecoeff':phiMinus2NS}
phiSDict = {'dchi0':phi0S, 'dchi1':phi1S, 'dchi2':phi2S, 'dchi3':phi3S, 'dchi4':phi4S, 'dchi5l':phi5lS, 'dchi6':phi6S, 'dchi6l':phi6lS, 'dchi7':phi7S, 'dchiminus2':phiMinus2S, 'dipolecoeff':phiMinus2S}

#Testing-GR parameter ranges used in BNS TGR paper
#
dchiminus2min = -1.0
dchiminus2max = 1.0
dchi0min = -5.0
dchi0max = 5.0
dchi1min = -5.0
dchi1max = 5.0
dchi2min = -5.0
dchi2max = 5.0
dchi3min = -5.0
dchi3max = 5.0
dchi4min = -10.0
dchi4max = 10.0
dchi5lmin = -5.0
dchi5lmax = 5.0
dchi6min = -5.0
dchi6max = 5.0
dchi6lmin = -20.0
dchi6lmax = 20.0
dchi7min = -30.0
dchi7max = 30.0

#Dictionaries that map the testing-GR parameter of each run to ranges function above
dchiMinDict = {'dchi0':dchi0min, 'dchi1':dchi1min, 'dchi2':dchi2min, 'dchi3':dchi3min, 'dchi4':dchi4min, 'dchi5l':dchi5lmin, 'dchi6':dchi6min, 'dchi6l':dchi6lmin, 'dchi7':dchi7min, 'dchiminus2':dchiminus2min, 'dipolecoeff':dchiminus2min}
dchiMaxDict = {'dchi0':dchi0max, 'dchi1':dchi1max, 'dchi2':dchi2max, 'dchi3':dchi3max, 'dchi4':dchi4max, 'dchi5l':dchi5lmax, 'dchi6':dchi6max, 'dchi6l':dchi6lmax, 'dchi7':dchi7max, 'dchiminus2':dchiminus2max, 'dipolecoeff':dchiminus2max}

def convert_SEOBNRT_to_PhenomPNRT_parameterization(data, param, bins_arg=25, nsamples=1000000):
  """Given a full set of posterior samples from a generic FD run, return the bins and PDF for dchi for an equivalent TIGER run"""

  if param in ['dchi0', 'dchi1', 'dchi2', 'dchi6l', 'dchiminus2']:
    return data[param]

  #First draw nsamples number of samples from the prior used in the runs for the BNS TGR paper.

  m1prior=random.uniform(compmin,compmax,nsamples)
  m2prior=random.uniform(compmin,compmax,nsamples)

  #The priors on z-component of spins are compatible with those used for runs with precession
  x1prior=random.uniform(-0.5,0.5,nsamples)
  x2prior=random.uniform(-0.5,0.5,nsamples)
  a1zprior=spinmax * np.real(-2.*x1prior/lambertw(-2.*np.abs(x1prior)/np.e,-1))
  a2zprior=spinmax * np.real(-2.*x2prior/lambertw(-2.*np.abs(x2prior)/np.e,-1))

  dchiprior_gFD = random.uniform(dchiMinDict[param],dchiMaxDict[param],nsamples)
  
  #Compute the prior distribution on dchi_i (parameterized as with TIGER) corresponding to a uniform distribution in dchi_i (parameterized as with generic FD)
  dchiprior_TIGER = []
  for i in range(len(dchiprior_gFD)):
    m1 = m1prior[i]
    m2 = m2prior[i]
    a1z = a1zprior[i]
    a2z = a2zprior[i]
    a1sq = a1zprior[i]*a1zprior[i]
    a2sq = a2zprior[i]*a2zprior[i]
    a1dota2 = a1zprior[i]*a2zprior[i]
    dchiprior_TIGER.append(dchiprior_gFD[i]*(1. + phiSDict[param](m1,m2,a1z,a2z,a1sq,a2sq,a1dota2)/phiNSDict[param](m1,m2)))
  
  #Convert the posetrior distribution of dchi_i (as parameterized with generic FD) into a distribution pf dchi_i (parameterized with TIGER)
  dchidata_TIGER = []
  for j in range(data.size):
    m1 = data['m1'][j]
    m2 = data['m2'][j]
    a1z = data['a1'][j] * data['costilt1'][j]
    a2z = data['a2'][j] * data['costilt2'][j]
    a1sq = data['a1'][j] * data['a1'][j]
    a2sq = data['a2'][j] * data['a2'][j]
    a1dota2 = data['a1'][j] * data['a2'][j] * data['costilt1'][j] * data['costilt2'][j]
    dchidata_TIGER.append(data[param][j]*(1. + phiSDict[param](m1,m2,a1z,a2z,a1sq,a2sq,a1dota2)/phiNSDict[param](m1,m2)))
  
  dchi_min=min(dchidata_TIGER)
  dchi_max=max(dchidata_TIGER)
  
  P_dchi_pr, dchi_bins = np.histogram(dchiprior_TIGER, bins=np.linspace(dchi_min,dchi_max,num=bins_arg+1), normed=True)
  P_dchi, dchi_bins = np.histogram(dchidata_TIGER, bins=dchi_bins, normed=True)
  P_dchi_gFD, dchi_bins = np.histogram(data[param], bins=dchi_bins, normed=True)
  
  #Compute the posterior distribution on dchi_i (as parameterized with TIGER) corresponding to a flat prior in dchi_i (as parameterized with TIGER) by reweighting by the prior on dchi_i (as parameterized with TIGER) given a flat prior on dchi_i (as parameterized by generic FD)
  bin_width=(dchi_bins[1]-dchi_bins[0])
  P_dchi_corrected = P_dchi/P_dchi_pr
  P_dchi_corrected[np.isnan(P_dchi_corrected)] = 0.
  P_dchi_corrected = P_dchi_corrected/(np.sum(P_dchi_corrected)*bin_width)
    
  dchi_bins_center = (dchi_bins[:-1]+dchi_bins[1:])/2.

  #Tabulate the discrete CDF, invert it, and interpolate the inverse CDF
  cumulative_values=np.zeros(dchi_bins.shape)
  cumulative_values[1:]=np.cumsum(P_dchi_corrected*np.diff(dchi_bins))
  inv_cdf = interpolate.interp1d(cumulative_values, dchi_bins)
  
  #Return the bins and values of the reweighted posterior distribution
  r = random.uniform(0.,1.,nsamples/10)
  
  return inv_cdf(r)

