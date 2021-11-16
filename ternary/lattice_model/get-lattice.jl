using DelimitedFiles

"""
function to initialize colored lattice

x_vec -> vector of volume fractions
dim -> lattice dimension (default = 100)
"""
function init_lattice(x_vec;dim=100)

    ## length of vector of volume fractions
    len = size(x_vec,1)
    ## init empty lattice
    lattice = zeros(Int,dim,dim)
    ## init vector of probability densities
    dens = [sum(x_vec[1:i]) for i=1:len]

    ## distribute colors on lattice according to their volume fraction
    for i = 1:dim
        for j = 1:dim
            rnd = rand()
            for k = 1:len
                if rnd < dens[k]
                    lattice[i,j] = k
                    break
                end
            end
        end
    end

    return lattice

end


"""
MC-metropolis style swap of two lattice sites

beta -> inverse temperature
M -> interaction matrix
lattice -> instant of a color lattice
dim -> lattice dimension (default = 100)
"""
function swap(beta,M,lattice;dim=100)
    ## select random lattice site 1
    i1,j1 = rand(1:dim),rand(1:dim)
    p1 = lattice[i1,j1]
    ## get neighbourhood
    pu1 = lattice[mod1(i1-1,dim),j1] ## up
    pd1 = lattice[mod1(i1+1,dim),j1] ## down
    pl1 = lattice[i1,mod1(j1-1,dim)] ## left
    pr1 = lattice[i1,mod1(j1+1,dim)] ## right
    ## get energy
    E1 = -(M[p1+1,pu1+1] + M[p1+1,pd1+1] + M[p1+1,pl1+1] + M[p1+1,pr1+1])

    ## select random lattice site 2
    i2,j2 = rand(1:dim),rand(1:dim)
    p2 = lattice[i2,j2]
    ## get neighbourhood
    pu2 = lattice[mod1(i2-1,dim),j2] ## up
    pd2 = lattice[mod1(i2+1,dim),j2] ## down
    pl2 = lattice[i2,mod1(j2-1,dim)] ## left
    pr2 = lattice[i2,mod1(j2+1,dim)] ## right
    ## get energy
    E2 = -(M[p2+1,pu2+1] + M[p2+1,pd2+1] + M[p2+1,pl2+1] + M[p2+1,pr2+1])

    ## get energies after swap
    En1 = -(M[p2+1,pu1+1] + M[p2+1,pd1+1] + M[p2+1,pl1+1] + M[p2+1,pr1+1])
    En2 = -(M[p1+1,pu2+1] + M[p1+1,pd2+1] + M[p1+1,pl2+1] + M[p1+1,pr2+1])

    ## get energy difference
    dE = En1 + En2 - (E1 + E2)

    ## acceptance probability
    if rand() < min(1,exp(-beta*dE))
        lattice[i1,j1] = p2
        lattice[i2,j2] = p1
    end
end


"""
sweep over lattice and apply mc-swap

n -> number of sweeps
beta -> inverse temperature
M -> interaction matrix
lattice -> instant of a color lattice
n_swaps -> nr of swaps performed in a sweep (default 10000)
"""
function sweep(n,beta,M,lattice;n_swaps=10000)
    @fastmath @inbounds for i = 1:n
        for j = 1:n_swaps
            swap(beta,M,lattice)
        end
    end
end


"""
main function

xb -> sovent 1 concentration (blue)
xr -> sovent 2 concentration (red)
"""
function main(xb,xr;n_sweep=1000000,dn=100,n_eq=100)

    if (xb+xr)<1.0
        
        dirname = "./lattice_dump/lattice-"*string(xb)*"-"*string(xr)*"/"

        if !isdir(dirname)
            mkdir(dirname)
        end

        # n_sweep = 1000000
        # dn = 100
        # n_eq = 100

        beta = 1.0
        Jww = 0.0; ## solvent-solvent interaction
        Jbb = -1.0; ## solute1-solute1 interaction
        Jrr = -1.0; ## solute2-solute2 interaction
        Jbw = 0.0; ## solvent-solute1 interaction
        Jrw = 0.0; ## solvent-solute2 interaction
        Jbr = 3.0; ## solute1-solute2 interaction

        Jbb = 1.0; Jrr = 1.0; Jbr = -3.0;
        # Jbb = 2.0; Jrr = 0.0; Jbr = 0.5;

        M = [Jww Jbw Jrw;
             Jbw Jbb Jbr;
             Jrw Jbr Jrr]; ## interaction matrix, M[i+1,j+1]
        lattice = init_lattice([xb,xr]);

        sweep(n_sweep-dn*n_eq,beta,M,lattice)

        for i=1:n_eq
            sweep(dn,beta,M,lattice)
            writedlm(dirname*"lattice-"*string(i)*".txt",lattice,',')
        end

    end

end

## execute main function with command line arguments as input
main(parse(Float64,ARGS[1]),parse(Float64,ARGS[2]))