using DelimitedFiles
using HDF5


"""
function that computes energy of lattice per site
requires as constants

L -> lattice size

arguments:
M -> interaction matrix
lattice -> instant of a color lattice
"""
function lattice_energy(M,lattice,dim)
    ## init energy
    E = 0.0
    ## loop over latice
    for i=1:dim
        for j=1:dim
            ## get state of lattice site i,j
            p = lattice[i,j]
            ## get neighbourhood
            pu = lattice[mod1(i-1,dim),j] ## up
            pd = lattice[mod1(i+1,dim),j] ## down
            pl = lattice[i,mod1(j-1,dim)] ## left
            pr = lattice[i,mod1(j+1,dim)] ## right
            ## get energy
            E += -(M[p+1,pu+1] + M[p+1,pd+1] + M[p+1,pl+1] + M[p+1,pr+1])
        end
    end
    ## return energy per lattice-side
    return E
end


"""
function to initialize colored lattice

x_vec -> vector of volume fractions
dim -> lattice dimension (default = 100)
"""
function init_lattice(x_vec,dim)

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
function swap(beta,M,lattice,dim)
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
function sweep(n,beta,M,lattice,dim;n_swaps=10000)
    @fastmath @inbounds for i = 1:n
        for j = 1:n_swaps
            swap(beta,M,lattice,dim)
        end
    end
end


"""
main function

xb -> sovent 1 concentration (blue)
xr -> sovent 2 concentration (red)
"""
function main(xb,xr;dim=100,n_therm=1000000,n_measure=1000,dt=1000,n_anneal=10000)

    if (xb+xr)<1.0
        
        fname = "./lattice_dump/lattice-"*string(xb)*"-"*string(xr)*".h5"
        fid = h5open(fname,"w")
        create_group(fid,"lattices")
        g = fid["lattices"]
        dset = create_dataset(g,"L",datatype(Int64),dataspace(dim,dim,n_measure),chunk=(dim,dim,1))
        create_group(fid,"energy")
        create_group(fid,"time")
        create_group(fid,"snapshots")

        beta = 1.0
        Jww = 0.0; ## solvent-solvent interaction
        Jbb = -1.0; ## solute1-solute1 interaction
        Jrr = -1.0; ## solute2-solute2 interaction
        Jbw = 0.0; ## solvent-solute1 interaction
        Jrw = 0.0; ## solvent-solute2 interaction
        Jbr = 3.0; ## solute1-solute2 interaction


        # Jbb = -1.0; Jrr = -1.0; Jbr = 3.0; # set 1 # associative
        # Jbb = 1.0; Jrr = 1.0; Jbr = -3.0; # set 2 # segregative case
        # Jbb = 2.0; Jrr = 0.0; Jbr = 0.5; # set 3 # counter-ionic
        
        M = [Jww Jbw Jrw;
             Jbw Jbb Jbr;
             Jrw Jbr Jrr]; ## interaction matrix, M[i+1,j+1]
        lattice = init_lattice([xb,xr],dim);

        if n_anneal != 0
            beta0 = 0.01
            k = (beta-beta0)/n_anneal
            d = beta0
            for t = 0:n0-1
                b = k*t+d
                sweep(1,b,M,lattice,dim)
            end
        end

        energy = [lattice_energy(M,lattice)]
        tsteps = [0]
        snapshots = Array{Int64,1}()

        for t = 1:n_therm
            sweep(1,beta,M,lattice,dim)
            push!(tsteps,t)
            push!(energy,lattice_energy(M,lattice))
        end

        count = 0
        for t = 1:n_measure*dt
            sweep(1,beta,M,lattice,dim)
            push!(energy,lattice_energy(M,lattice))
            count += 1
            if count == dt
                dset[:,:,Int(t/dt)] = lattice
                push!(snapshots,n_therm+t)
                count = 0
            end
            push!(tsteps,n_therm+t)
        end

        fid["energy/E"] = energy
        fid["time/tsteps"] = tsteps
        fid["snapshots/s"] = snapshots

        close(fid)

    end

end

## execute main function with command line arguments as input
main(parse(Float64,ARGS[1]),parse(Float64,ARGS[2]))