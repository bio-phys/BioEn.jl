module InputOutput 

using DelimitedFiles #, HDF5
# input

## txt files

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
end

function write_txt_normalized(path; Y_name, Y, y_name, y, w0_name="", w0=[])
    write_txt(path*Y_name, Y)
    write_txt(path*y_name, y)
    if w0_name != ""
         write_txt(path*w0_name, w0)
    end
end


# function write(y)
# end


## txt files

## HDF5 files

# for (key, val) in output
#    h5write(oname, key, val)
# end

end # end module IO
