""" This file contains classical code in support of Shor's Algorithm
"""

""" Imports to Python functions """
import math
import array
import fractions
import numpy as np

import sys


""" ******** Pre-Quantum *** """

""" Function to check if N is of type q^p"""
def check_if_power(N):
    """ Check if N is a perfect power in O(n^3) time, n=ceil(logN) """
    b=2
    while (2**b) <= N:
        a = 1
        c = N
        while (c-a) >= 2:
            m = int( (a+c)/2 )

            if (m**b) < (N+1):
                p = int( (m**b) )
            else:
                p = int(N+1)

            if int(p) == int(N):
                print('N is {0}^{1}'.format(int(m),int(b)) )
                return True

            if p<N:
                a = int(m)
            else:
                c = int(m)
        b=b+1

    return False


""" ******** Post-Quantum *** """

""" Function to get the value a ( 1<a<N ), such that a and N are coprime. Starts by getting the smallest a possible
    This normally can be done fully randomly, it is done like this for the user to have control 
    over the a value that gets selected """
def get_value_a(N):

    """ ok defines if user wants to used the suggested a (if ok!='0') or not (if ok=='0') """
    ok='0'

    """ Starting with a=2 """
    a=2

    """ Get the smallest a such that a and N are coprime"""
    while math.gcd(a,N)!=1:
        a=a+1

    """ Store it as the smallest a possible """
    smallest_a = a

    """ Ask user if the a found is ok, if not, then increment and find the next possibility """  
    ok = input('Is the number {0} ok for a? Press 0 if not, other number if yes: '.format(a))
    if ok=='0':
        if(N==3):
            print('Number {0} is the only one you can use. Using {1} as value for a\n'.format(a,a))
            return a
        a=a+1

    """ Cycle to find all possibilities for a not counting the smallest one, until user says one of them is ok """
    while ok=='0':
        
        """ Get a coprime with N """
        while math.gcd(a,N)!=1:
            a=a+1
    
        """ Ask user if ok """
        ok = input('Is the number {0} ok for a? Press 0 if not, other number if yes: '.format(a))

        """ If user says it is ok, then exit cycle, a has been found """
        if ok!='0':
            break
        
        """ If user says it is not ok, increment a and check if are all possibilites checked.  """
        a=a+1

        """ If all possibilities for a are rejected, put a as the smallest possible value and exit cycle """
        if a>(N-1):
            print('You rejected all options for value a, selecting the smallest one\n')
            a=smallest_a
            break

    """ Print the value that is used as a """
    print('Using {0} as value for a\n'.format(a))

    return a
    

""" Function to apply the continued fractions to find r and the gcd to find the desired factors"""
def get_factors(x_value,t_upper,N,a):

    if x_value<=0:
        print('x_value is <= 0, there are no continued fractions\n')
        return False

    print('Running continued fractions for this case\n')

    """ Calculate T and x/T """
    T = pow(2,t_upper)

    x_over_T = x_value/T

    """ Cycle in which each iteration corresponds to putting one more term in the
    calculation of the Continued Fraction (CF) of x/T """

    """ Initialize the first values according to CF rule """
    i=0
    b = array.array('i')
    t = array.array('f')

    b.append(math.floor(x_over_T))
    t.append(x_over_T - b[i])

    while i>=0:

        """From the 2nd iteration onwards, calculate the new terms of the CF based
        on the previous terms as the rule suggests"""

        if i>0:
            b.append( math.floor( 1 / (t[i-1]) ) ) 
            t.append( ( 1 / (t[i-1]) ) - b[i] )

        """ Calculate the CF using the known terms """

        aux = 0
        j=i
        while j>0:    
            aux = 1 / ( b[j] + aux )      
            j = j-1
        
        aux = aux + b[0]

        """Get the denominator from the value obtained"""
        frac = fractions.Fraction(aux).limit_denominator()
        den=frac.denominator

        print('Approximation number {0} of continued fractions:'.format(i+1))
        print("Numerator:{0} \t\t Denominator: {1}\n".format(frac.numerator,frac.denominator))

        """ Increment i for next iteration """
        i=i+1

        if (den%2) == 1:
            if i>=15:
                print('Returning because have already done too much tries')
                return False
            print('Odd denominator, will try next iteration of continued fractions\n')
            continue
    
        """ If denominator even, try to get factors of N """

        """ Get the exponential a^(r/2) """

        exponential = 0

        if den<1000:
            exponential=pow(a , (den/2))
        
        """ Check if the value is too big or not """
        if math.isinf(exponential)==1 or exponential>1000000000:
            print('Denominator of continued fraction is too big!\n')
            aux_out = input('Input number 1 if you want to continue searching, other if you do not: ')
            if aux_out != '1':
                return False
            else:
                continue

        """If the value is not to big (infinity), then get the right values and
        do the proper gcd()"""

        putting_plus = int(exponential + 1)

        putting_minus = int(exponential - 1)
    
        one_factor = math.gcd(putting_plus,N)
        other_factor = math.gcd(putting_minus,N)
    
        """ Check if the factors found are trivial factors or are the desired
        factors """

        if one_factor==1 or one_factor==N or other_factor==1 or other_factor==N:
            print('Found just trivial factors, not good enough\n')
            """ Check if the number has already been found, use i-1 because i was already incremented """
            if t[i-1]==0:
                print('The continued fractions found exactly x_final/(2^(2n)) , leaving funtion\n')
                return False
            if i<15:
                aux_out = input('Input number 1 if you want to continue searching, other if you do not: ')
                if aux_out != '1':
                    return False       
            else:
                """ Return if already too much tries and numbers are huge """ 
                print('Returning because have already done too many tries\n')
                return False         
        else:
            print('The factors of {0} are {1} and {2}\n'.format(N,one_factor,other_factor))
            print('Found the desired factors!\n')
            return True
