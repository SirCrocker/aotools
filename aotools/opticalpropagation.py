'''
A library of useful optical propagation methods.

Many extracted from the book by Schmidt, 2010: Numerical Methods
of optical proagation
'''

import numpy
from . import fouriertransform


def bandLimitedAngularSpectrum(inputComplexAmp, wvl, inputSpacing, outputSpacing, z):
    """
    Propogates light complex amplitude using an angular spectrum algorithm. The algorithm
    band limits the propagating signal to ((reduce aliasing and reduce numerical errors))

    Parameters:
        inputComplexAmp (ndarray): Complex array of input complex amplitude
        wvl (float): Wavelength of light to propagate
        inputSpacing (float): The spacing between points on the input array in metres
        outputSpacing (float): The desired spacing between points on the output array in metres
        z (float): Distance to propagate in metres

    Returns:
        ndarray: propagated complex amplitude
    """


    # Band limit auxilary function
    def rect(array):
        """Rectangle function.

        Args:
            array (numpy.array): an array of values

        Returns:
            numpy.array: 1 where abs(array) <= 0.5 0 otherwise
        """
        # we are not setting the value as 0.5 at the borders
        return numpy.where(abs(array) <= 0.5, 1 ,0) 

    
    # If propagation distance is 0, don't bother 
    if z==0:
        return inputComplexAmp

    N_og = inputComplexAmp.shape[0] #Assumes Uin is square and N_og is even.
    k = 2*numpy.pi/wvl     #optical wavevector

    # We double the sampling windows and pad with 0 
    # circ conv to linear conv
    inputComplexAmp = numpy.pad(
        inputComplexAmp,
        [(N_og//2, N_og//2), (N_og//2, N_og//2)],
        constant_values=0
    )

    N = 2 * N_og

    (x1,y1) = numpy.meshgrid(inputSpacing*numpy.arange(-N/2,N/2),
                             inputSpacing*numpy.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10

    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*inputSpacing)
    fX,fY = numpy.meshgrid(df1*numpy.arange(-N/2,N/2),
                           df1*numpy.arange(-N/2,N/2))
    fsq = fX**2 + fY**2

    #Scaling Param
    mag = float(outputSpacing)/inputSpacing

    #Observation Plane Co-ords
    x2,y2 = numpy.meshgrid( outputSpacing*numpy.arange(-N/2,N/2),
                            outputSpacing*numpy.arange(-N/2,N/2) )
    r2sq = x2**2 + y2**2

    #Quadratic phase factors
    Q1 = numpy.exp( 1j * k/2. * (1-mag)/z * r1sq) # spatial domain

    Q2 = numpy.exp(-1j * numpy.pi**2 * 2 * z/mag/k*fsq) # transfer function / frequency domain

    fX_limit = 1 / (numpy.sqrt( numpy.pow(2 * df1 * z, 2) + 1 ) * wvl)
    fY_limit = fX_limit
    Q2_BL = Q2 * rect(fX / (2 * fX_limit)) * rect(fY / (2 * fY_limit)) # band limited

    Q3 = numpy.exp(1j * k/2. * (mag-1)/(mag*z) * r2sq) # spatial domain

    #Compute propagated field
    outputComplexAmp = Q3 * fouriertransform.ift2(
                    Q2_BL * fouriertransform.ft2(Q1 * inputComplexAmp/mag,inputSpacing), df1)

    # We slice back to the original values
    sl_b = N_og // 2
    sl_e = N_og // 2 * 3
    outputComplexAmp = outputComplexAmp[sl_b:sl_e, sl_b:sl_e]

    return outputComplexAmp


def angularSpectrum(inputComplexAmp, wvl, inputSpacing, outputSpacing, z):
    """
    Propogates light complex amplitude using an angular spectrum algorithm

    Parameters:
        inputComplexAmp (ndarray): Complex array of input complex amplitude
        wvl (float): Wavelength of light to propagate
        inputSpacing (float): The spacing between points on the input array in metres
        outputSpacing (float): The desired spacing between points on the output array in metres
        z (float): Distance to propagate in metres

    Returns:
        ndarray: propagated complex amplitude
    """
    
    # If propagation distance is 0, don't bother 
    if z==0:
        return inputComplexAmp

    N = inputComplexAmp.shape[0] #Assumes Uin is square.
    k = 2*numpy.pi/wvl     #optical wavevector

    (x1,y1) = numpy.meshgrid(inputSpacing*numpy.arange(-N/2,N/2),
                             inputSpacing*numpy.arange(-N/2,N/2))
    r1sq = (x1**2 + y1**2) + 1e-10

    #Spatial Frequencies (of source plane)
    df1 = 1. / (N*inputSpacing)
    fX,fY = numpy.meshgrid(df1*numpy.arange(-N/2,N/2),
                           df1*numpy.arange(-N/2,N/2))
    fsq = fX**2 + fY**2

    #Scaling Param
    mag = float(outputSpacing)/inputSpacing

    #Observation Plane Co-ords
    x2,y2 = numpy.meshgrid( outputSpacing*numpy.arange(-N/2,N/2),
                            outputSpacing*numpy.arange(-N/2,N/2) )
    r2sq = x2**2 + y2**2

    #Quadratic phase factors
    Q1 = numpy.exp( 1j * k/2. * (1-mag)/z * r1sq)

    Q2 = numpy.exp(-1j * numpy.pi**2 * 2 * z/mag/k*fsq)

    Q3 = numpy.exp(1j * k/2. * (mag-1)/(mag*z) * r2sq)

    #Compute propagated field
    outputComplexAmp = Q3 * fouriertransform.ift2(
                    Q2 * fouriertransform.ft2(Q1 * inputComplexAmp/mag,inputSpacing), df1)
    return outputComplexAmp


def oneStepFresnel(Uin, wvl, d1, z):
    """
    Fresnel propagation using a one step Fresnel propagation method.

    Parameters:
        Uin (ndarray): A 2-d, complex, input array of complex amplitude
        wvl (float): Wavelength of propagated light in metres
        d1 (float): spacing of input plane
        z (float): metres to propagate along optical axis

    Returns:
        ndarray: Complex ampltitude after propagation
    """
    N = Uin.shape[0]    #Assume square grid
    k = 2*numpy.pi/wvl  #optical wavevector

    #Source plane coordinates
    x1,y1 = numpy.meshgrid( numpy.arange(-N/2.,N/2.) * d1,
                            numpy.arange(-N/2.,N/2.) * d1)
    #observation plane coordinates
    d2 = wvl*z/(N*d1)
    x2,y2 = numpy.meshgrid( numpy.arange(-N/2.,N/2.) * d2,
                            numpy.arange(-N/2.,N/2.) * d2 )

    #evaluate Fresnel-Kirchoff integral
    A = 1/(1j*wvl*z)
    B = numpy.exp( 1j * k/(2*z) * (x2**2 + y2**2))
    C = fouriertransform.ft2(Uin *numpy.exp(1j * k/(2*z) * (x1**2+y1**2)), d1)

    Uout = A*B*C

    return Uout

def twoStepFresnel(Uin, wvl, d1, d2, z):
    """
    Fresnel propagation using a two step Fresnel propagation method.

    Parameters:
        Uin (ndarray): A 2-d, complex, input array of complex amplitude
        wvl (float): Wavelength of propagated light in metres
        d1 (float): spacing of input plane
        d2 (float): desired output array spacing
        z (float): metres to propagate along optical axis

    Returns:
        ndarray: Complex ampltitude after propagation
    """

    N = Uin.shape[0] #Number of grid points
    k = 2*numpy.pi/wvl #optical wavevector

    #source plane coordinates
    x1, y1 = numpy.meshgrid( numpy.arange(-N/2,N/2) * d1,
                            numpy.arange(-N/2.,N/2.) * d1 )

    #magnification
    m = float(d2)/d1

    #intermediate plane
    try:
        Dz1  = z / (1-m) #propagation distance
    except ZeroDivisionError:
        Dz1 = z / (1+m)
    d1a = wvl * abs(Dz1) / (N*d1) #coordinates
    x1a, y1a = numpy.meshgrid( numpy.arange( -N/2.,N/2.) * d1a,
                              numpy.arange( -N/2.,N/2.) * d1a )

    #Evaluate Fresnel-Kirchhoff integral
    A = 1./(1j * wvl * Dz1)
    B = numpy.exp(1j * k/(2*Dz1) * (x1a**2 + y1a**2) )
    C = fouriertransform.ft2(Uin * numpy.exp(1j * k/(2*Dz1) * (x1**2 + y1**2)), d1)
    Uitm = A*B*C
    #Observation plane
    Dz2 = z - Dz1

    #coordinates
    x2,y2 = numpy.meshgrid( numpy.arange(-N/2., N/2.) * d2,
                            numpy.arange(-N/2., N/2.) * d2 )

    #Evaluate the Fresnel diffraction integral
    A = 1. / (1j * wvl * Dz2)
    B = numpy.exp( 1j * k/(2 * Dz2) * (x2**2 + y2**2) )
    C = fouriertransform.ft2(Uitm * numpy.exp( 1j * k/(2*Dz2) * (x1a**2 + y1a**2)), d1a)
    Uout = A*B*C

    return Uout

def lensAgainst(Uin, wvl, d1, f):
    '''
    Propagates from the pupil plane to the focal
    plane for an object placed against (and just before)
    a lens.

    Parameters:
        Uin (ndarray): Input complex amplitude
        wvl (float): Wavelength of light in metres
        d1 (float): spacing of input plane
        f (float): Focal length of lens

    Returns:
        ndarray: Output complex amplitude
    '''

    N = Uin.shape[0] #Assume square grid
    k = 2*numpy.pi/wvl  #Optical Wavevector

    #Observation plane coordinates
    fX = numpy.arange( -N/2.,N/2.)/(N*d1)

    #Observation plane coordinates
    x2,y2 = numpy.meshgrid(wvl * f * fX, wvl * f * fX)
    del(fX)

    #Evaluate the Fresnel-Kirchoff integral but with the quadratic
    #phase factor inside cancelled by the phase of the lens
    Uout = numpy.exp( 1j*k/(2*f) * (x2**2 + y2**2) )/ (1j*wvl*f) * fouriertransform.ft2( Uin, d1)

    return Uout
