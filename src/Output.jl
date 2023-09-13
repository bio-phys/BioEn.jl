"""
Writing out results of BioEn refinemnt. 
"""
module Output

using HDF5 # for compatibility with Python 

function delete(filename)
    if isfile(filename) 
        println("Deleting file \"$filename\".")
        rm(filename)
    else
        println("File \"$filename\" does not exist.")
    end
end

function write_HDF5(filename; ws, thetas)
    h5write(filename, "ws", ws)
    h5write(filename, "thetas", thetas)
end

function read_HDF5(filename)
    ws = h5read(filename, "ws")
    thetas = h5read(filename, "thetas")
end


end # end module Output 
