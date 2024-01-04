"""
Writing out results of BioEn refinemnt. 
"""
module Output

using HDF5 # for compatibility with Python 

function delete(filename)
    if isfile(filename) 
        println("\nDeleting file \"$filename\".")
        rm(filename)
    else
        println("\nFile \"$filename\" does not exist. No deletion necessary.")
    end
    return
end

function write_HDF5(filename; ws, thetas)
    println("\nWriting file \"$filename\".")
    h5write(filename, "ws", ws)
    h5write(filename, "thetas", thetas)
    return
end

function read_HDF5(filename)
    println("\nReading file \"$filename\".")
    ws = h5read(filename, "ws")
    thetas = h5read(filename, "thetas")
    return ws, thetas
end


end # end module Output 
