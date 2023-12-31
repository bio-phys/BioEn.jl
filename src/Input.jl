"""
Reading in input data for BioEn refinement. 

Input data consists of (experimental) data, calculated observables, and optionally reference weights. 
"""
#todo: errors should be provided optionally 
module Input

using DelimitedFiles #, HDF5
# input

export read_txt_normalized, write_txt_normalized

## txt files

function modify_path(path)
    if path[end]!='/'
        path=path*"/"
    end
    return path
end

function have_consistent_sizes(Y::Vector{T}, y::Array{T, 2}) where T<:AbstractFloat
    MY = size(Y,1)
    My = size(y, 2) 
    MY == My && return true
    #error("Shapes of data (M=$MY) and calculated observables (M=$My) are not consistent!")
    println("Shapes of data (M=$MY) and calculated observables (M=$My) are not consistent!")
    return false
end

function have_consistent_sizes(y::Array{T,2}, w0) where T<:AbstractFloat
    if w0 == [] 
        return true
    else
        Ny = size(y,1)
        Nw0 = size(w0, 1)
        Ny == Nw0 && return true
        println("Shapes of calculated observables (N=$Ny) and reference weights (N=$Nw0) are not consistent!")  
        return false
    end
end

function have_consistent_sizes(Y::Vector{T}, y::Array{T,2}, w0) where T<:AbstractFloat
    return have_consistent_sizes(Y, y) && have_consistent_sizes(y, w0) 
end

# function consistent_eltype()???

function read_txt(filename)
    data = DelimitedFiles.readdlm(filename)
    return data, size(data)
end

function read_txt_1D(filename)
    data, ssize = read_txt(filename)
    return vec(data), ssize[1]
end

function read_txt_normalized(path; Y_name, y_name, w0_name="")
    println("Reading files from path \"$path\".\n")
    println("Reading data from file \"$Y_name\".")
    path = modify_path(path)
    Y, MY = read_txt_1D(path*Y_name)
    println("Reading calculated observables from file \"$y_name\".")
    y, (Ny, My) = read_txt(path*y_name)
    MY != My && error("Different numbers of calculated (M=$My from file \"$y_name\") and measured (M=$MY from file \"$Y_name\") observables!")
    if w0_name == ""
        println("Generating N=$Ny uniform reference weights.")
        w0 = ones(Ny)./Ny
    else
        println("Reading reference weights from file \"$w0_name\".")
        w0, Nw0 = read_txt_1D(path*w0_name)
        Ny != Nw0 && error("Different ensemble sizes for calculated observables (N=$Ny from file \"$y_name\") and reference weights (N=$Nw0 from file \"$w0_name\")!")
    end
    return Y, y, w0, Ny, My
end

# output 

function write_txt(filename, arr)
    DelimitedFiles.writedlm(filename, arr)
    return 
end

function write_txt_normalized(path; Y_name, Y, y_name, y, w0_name="", w0=[])
    if have_consistent_sizes(Y, y, w0) 
        println("Writing files to path \"$path\".\n")
        println("Writing data to file \"$Y_name\".")
        path = modify_path(path)
        write_txt(path*Y_name, Y)
        println("Writing calculated observables to file \"$y_name\".")
        write_txt(path*y_name, y)
        if w0_name != ""
            println("Writing reference weights to file \"$w0_name\".")
            write_txt(path*w0_name, w0)
        end
    else
        println("Shapes of arrays are not consistent! No output!")
    end
    return
end


# function write(y)
# end


## txt files

## HDF5 files

# for (key, val) in output
#    h5write(oname, key, val)
# end

end # end module IO
