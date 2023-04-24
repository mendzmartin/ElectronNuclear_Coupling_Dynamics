# Reference: https://gist.github.com/dataPulverizer/3dc0af456a427aeb704a437e31299242
# run command = julia BinaryFileIO_testing.jl

# Write binary file (Array)
function write_bin(x::Array{T, 1}, fileName::String)::Int64 where T 
    # Open the file
    io = open(fileName,"w")
    # Cast this number to make sure we know its type
    write(io, Int64(size(x)[1]))
    # Get the type as a string
    typ = repr(T)
    # Write the length of the type string
    write(io, Int64(length(typ)))
    # Now write the type string
    for i in eachindex(typ)
        write(io, Char(typ[i]))
    end
    # Now write the array
    for i in eachindex(x)
        write(io, x[i])
    end
    # Clean up
    close(io)
    return 0;
end

# Write binary file (Matrix)
function write_bin(x::Matrix{T}, fileName::String)::Int64 where T 
    # Open the file
    io = open(fileName,"w")
    # Cast this number to make sure we know its type
    write(io, Int64(size(x)[1]))
    # Get the type as a string
    typ = repr(T)
    # Write the length of the type string
    write(io, Int64(length(typ)))
    # Now write the type string
    for i in eachindex(typ)
        write(io, Char(typ[i]))
    end
    # Now write the array
    for i in eachindex(x)
        write(io, x[i])
    end
    # Clean up
    close(io)
    return 0;
end

# Read binary file
# Sub Function which speeds up the read
function read_bin(io::IO, ::Type{T}, n::Int64, matrix_data) where T
    # The array to be returned
    (matrix_data==true) ? x = Matrix{T}(undef, (n,n)) : x = Array{T, 1}(undef, n)
    @time for i in eachindex(x)
        x[i] = read(io, T)
    end
    close(io)
    return x
end

# The read function
function read_bin(fileName::String; matrix_data=false)
    # Open the file
    io = open(fileName, "r")
    # Read the total number of elements in the resulting array
    n = read(io, Int64)
    # Read the length of the type name
    nt = read(io, Int64)
    # Then read the type name
    cName = Array{Char}(undef, nt)
    for i in eachindex(cName)
        cName[i] = read(io, Char)
    end
    # The return type
    T = eval(Symbol(String(cName)))
    # The data
    x = read_bin(io, T, n , matrix_data)
    return x
end

function read_bin(fileName::String, ::Type{T}; matrix_data=false) where T
    # Open the file
    io = open(fileName, "r")
    # Read the total number of elements in the resulting array
    n = read(io, Int64)
    # Read the length of the type name
    nt = read(io, Int64)
    # Then read the type name
    cName = Array{Char}(undef, nt)
    for i in eachindex(cName)
        cName[i] = read(io, Char)
    end
    # The array to be returned
    (matrix_data==true) ? x = Matrix{T}(undef, (n,n)) : x = Array{T, 1}(undef, n)
    @time for i in eachindex(x)
        x[i] = read(io, T)
    end
    close(io)
    return x
end

function write_data(data,outfile_name;delim=" ",matrix_data=false,existing_file=false)
    if existing_file
        rm(outfile_name)
    end
    if matrix_data
        for f in 1:length(data[:,1])
            open(outfile_name, "a") do io
                writedlm(io,[data[f,:]]," ")
		flush(io)
            end
        end
    else
        for f in 1:length(data)
            open(outfile_name, "a") do io
                writedlm(io,[data[f]]," ")
		flush(io)
            end
        end
    end
end

begin
    # Warm up
    n = 1000;
    data1 = rand(Float64, (n,n));
    println("Size of data = $(sizeof(data1))")

    binFile = "data.bin"
    @time write_bin(data1, binFile);
    # data2 = read_bin(binFile,matrix_data=true);  # better reading
    # data3 = read_bin(binFile, eltype(data1));   # worst reading
    println("First method (better) -> size=$(filesize(binFile)/1e6)[MB]")
    rm(binFile)

    using DelimitedFiles;
    binFile = "data_02.dat"
    # Timed write read
    @time write_data(data1,binFile;matrix_data=true);
    println("Second method (worst) -> size=$(filesize(binFile)/1e6)[MB]")
    rm(binFile)
end