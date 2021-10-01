""" This file allows to test the calculations done before creating the quantum circuit
It asks the user for a number N and then:
    Checks if N is 1 or 0 or is an even -> these cases are simple
    Checks if N can be put in q^p form, q and p integers -> this can be done quicker classicaly then using the quantum circuit
If it is not an easy case like the above ones, then the program gets an integer a, coprime with N, starting with the 
smallest one possible and asking the user if the selected a is ok, if not, going to the second smallest one, and like this until
the user agrees with an a
"""

from cfunctions import check_if_power, get_value_a

""" Main program """
if __name__ == '__main__':

    """ Ask for analysis number N """   

    N = int(input('Please insert integer number N: '))

    print('input number was: {0}\n'.format(N))

    """ Check if N==1 or N==0"""

    if N==1 or N==0: 
       print('Please put an N different from 0 and from 1')
       exit()

    """ Check if N is even """

    if (N%2)==0:
        print('N is even, so does not make sense!')
        exit()
    
    """ Check if N can be put in N=p^q, p>1, q>=2 """

    """ Try all numbers for p: from 2 to sqrt(N) """
    if check_if_power(N)==True:
       exit()

    print('Not an easy case, using the quantum circuit is necessary\n')

    a=get_value_a(N)