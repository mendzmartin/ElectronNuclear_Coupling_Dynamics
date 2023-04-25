# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# módulo para construir grilla (1D)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
name_code = "Testing_ElectronNuclearDynamics_BinaryIOData_ScaleNuclearCoord";
#import Pkg;Pkg.resolve();Pkg.instantiate();Pkg.precompile()
include("../modules/module_schrodinger_equation_eigenproblem.jl");
include("./BinaryFileIO_testing.jl");

# run command = julia -O3 -t 4 Testing_ElectronNuclearDynamics_BinaryIOData_ScaleNuclearCoord.jl > Output_Testing_ElectronNuclearDynamics_BinaryIOData_ScaleNuclearCoord.log

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Creamos funciones útiles
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function Trapezoidal_Integration_Method(x_vec,fx_vec)
    dim_x=length(x_vec);
    coef_vec=ones(dim_x);
    coef_vec[2:(dim_x-1)].=2.0;
    @views function_vec=copy(fx_vec);
    Δx=abs(x_vec[2]-x_vec[1]); # válido para cuando Δx es constante
    return 0.5*Δx*dot(coef_vec,function_vec);
end

function integration_argument_diff_shannon_entropy(ρ_x_vector)
    ρlogρ_vec=similar(ρ_x_vector);
    Threads.@threads for index in eachindex(ρ_x_vector)
        ρ_x_vector[index]==0.0 ? ρlogρ_vec[index]=0.0 : ρlogρ_vec[index]=ρ_x_vector[index]*log(ρ_x_vector[index])
    end
    return ρlogρ_vec
end

function Reduced_TimeDependent_Diff_Shannon_Entropy(x_vec,ρ_x_matrix)
    Sx_vector=similar(ρ_x_matrix[1,:]);
    Threads.@threads for i in eachindex(Sx_vector)
        Sx_vector[i]=Trapezoidal_Integration_Method(x_vec,integration_argument_diff_shannon_entropy(ρ_x_matrix[:,i]));
    end
    return -1.0 .* Sx_vector;
end

function create_initial_state_2D(params;TypeOfFunction="FunctionScalingVariable")
    if TypeOfFunction=="FunctionScalingVariable"
        χ₀,β,ϕₙᵢ,Ω,dΩ = params;
        𝛹ₓ₀ = CellField(x->exp(-β*pow((x[2]*(1.0/γ)-χ₀),2)),Ω)*ϕₙᵢ;
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    elseif TypeOfFunction=="OriginalFunction"
        χ₀,β,ϕₙᵢ,Ω,dΩ = params;
        𝛹ₓ₀ = CellField(x->exp(-β*pow((x[2]-χ₀),2)),Ω)*ϕₙᵢ;
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    elseif TypeOfFunction=="OriginalFunctionBOAprox_v1"
        χ₀,β,ϕₙᵢ,Ω,dΩ = params;
        𝛹ₓ₀ = CellField(x->exp(-β*pow((x[2]-χ₀),2))*ϕₙᵢ(Point(x[1],χ₀)),Ω);
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    elseif TypeOfFunction=="OriginalFunctionBOAprox_v2"
        χ₀,β,ϕₙᵢ,Ω,dΩ,TrialSpace = params;
        𝛹ₓ₀Gridap = CellField(x->exp(-β*pow((x[2]-χ₀),2))*ϕₙᵢ(Point(x[1],χ₀)),Ω);
        𝛹ₓ₀ = interpolate_everywhere(𝛹ₓ₀Gridap,TrialSpace);
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    elseif TypeOfFunction=="OriginalFunctionBOAprox_v3"
        χ₀,β,ϕₙᵢInterpolated,Ω,dΩ,TrialSpace = params;
        𝛹ₓ₀Gridap = CellField(x->exp(-β*pow((x[2]-χ₀),2))*ϕₙᵢInterpolated(x[1]),Ω);
        𝛹ₓ₀ = interpolate_everywhere(𝛹ₓ₀Gridap,TrialSpace);
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    elseif TypeOfFunction=="OriginalFunctionBOAprox_v4"
        χ₀,β,ϕₙᵢ,Ω,dΩ,TrialSpace = params;
        𝛹ₓ₀Gridap = CellField(x->exp(-β*pow((x[2]-χ₀),2))*ϕₙᵢ(Point(x[1])),Ω);
        𝛹ₓ₀ = interpolate_everywhere(𝛹ₓ₀Gridap,TrialSpace);
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    elseif TypeOfFunction=="OriginalFunctionBOAprox_v5"
        χ₀,β,ϕₙᵢ,Ω,dΩ,TrialSpace = params;
        𝛹ₓ₀Gridap = CellField(x->exp(-β*pow((x[2]*(1.0/γ)-χ₀),2))*ϕₙᵢ(Point(x[1])),Ω);
        𝛹ₓ₀ = interpolate_everywhere(𝛹ₓ₀Gridap,TrialSpace);
        𝛹ₓ₀ = 𝛹ₓ₀*(1.0/norm_L2(𝛹ₓ₀,dΩ));
    end
    return 𝛹ₓ₀;
end

function Partial_probability_density(𝛹ₓ_vector,x₁_vector,x₂_vector,TrialSpace,Ω,dΩ;TypeAproxDeltaFunction="StepFunction",Improved=false)

    ρ_x₁_matrix=zeros(Float64,length(x₁_vector),length(𝛹ₓ_vector));
    ρ_x₂_matrix=zeros(Float64,length(x₂_vector),length(𝛹ₓ_vector));

    N₁=abs(x₁_vector[end]-x₁_vector[1]);
    N₂=abs(x₂_vector[end]-x₂_vector[1]);

    if (TypeAproxDeltaFunction=="StepFunction")
        Δx₁=abs(x₁_vector[2]-x₁_vector[1]);
        Δx₂=abs(x₂_vector[2]-x₂_vector[1]);
    end

    Threads.@threads for t_index in eachindex(𝛹ₓ_vector)
        𝛹ₓᵢ=interpolate_everywhere(𝛹ₓ_vector[t_index],TrialSpace);
        𝛹ₓᵢ=𝛹ₓᵢ/norm_L2(𝛹ₓᵢ,dΩ);
        ρₓᵢ=real(𝛹ₓᵢ'*𝛹ₓᵢ);

        Threads.@threads for x₁_index in eachindex(x₁_vector)
            if (TypeAproxDeltaFunction=="StepFunction")
                params=(x₁_vector[x₁_index],1.0,1,Δx₁)
            elseif (TypeAproxDeltaFunction=="BumpFunction")
                params=(x₁_vector[x₁_index],1.0,1)
            end
            δKroneckerGridap=CellField(x->AproxDiracDeltaFunction(x,params;TypeFunction=TypeAproxDeltaFunction),Ω);
            δnorm=sum(integrate(δKroneckerGridap,dΩ));
            if (TypeAproxDeltaFunction=="StepFunction")
                params=(x₁_vector[x₁_index],δnorm/N₂,1,Δx₁)
            elseif (TypeAproxDeltaFunction=="BumpFunction")
                params=(x₁_vector[x₁_index],δnorm/N₂,1)
            end
            δKroneckerGridap=CellField(x->AproxDiracDeltaFunction(x,params;TypeFunction=TypeAproxDeltaFunction),Ω);
            Improved==true ? ρ_x₁_matrix[t_index,x₁_index]=sum(integrate(ρₓᵢ*δKroneckerGridap,dΩ)) : ρ_x₁_matrix[x₁_index,t_index]=sum(integrate(ρₓᵢ*δKroneckerGridap,dΩ))
        end

        Threads.@threads for x₂_index in eachindex(x₂_vector)
            if (TypeAproxDeltaFunction=="StepFunction")
                params=(x₂_vector[x₂_index],1.0,2,Δx₂)
            elseif (TypeAproxDeltaFunction=="BumpFunction")
                params=(x₂_vector[x₂_index],1.0,2)
            end
            δKroneckerGridap=CellField(x->AproxDiracDeltaFunction(x,params;TypeFunction=TypeAproxDeltaFunction),Ω);
            δnorm=sum(integrate(δKroneckerGridap,dΩ));
            if (TypeAproxDeltaFunction=="StepFunction")
                params=(x₂_vector[x₂_index],δnorm/N₁,2,Δx₂)
            elseif (TypeAproxDeltaFunction=="BumpFunction")
                params=params=(x₂_vector[x₂_index],δnorm/N₁,2)
            end
            δKroneckerGridap=CellField(x->AproxDiracDeltaFunction(x,params;TypeFunction=TypeAproxDeltaFunction),Ω);
            Improved==true ? ρ_x₂_matrix[t_index,x₂_index]=sum(integrate(ρₓᵢ*δKroneckerGridap,dΩ)) : ρ_x₂_matrix[x₂_index,t_index]=sum(integrate(ρₓᵢ*δKroneckerGridap,dΩ))
        end
    end
    return ρ_x₁_matrix,ρ_x₂_matrix;
end

function position_expectation_value(𝛹ₓₜ,Ω,dΩ,TrialSpace,x_component)
    xGridap=CellField(x->x[x_component],Ω);
    x_ExpValue_vector=zeros(Float64,length(𝛹ₓₜ));
    Threads.@threads for time_index in eachindex(𝛹ₓₜ)
        𝛹ₓₜⁱ=interpolate_everywhere(𝛹ₓₜ[time_index],TrialSpace)
        # ojo! tomamos la parte real porque se trata de la coord. espacial, pero puede ser complejo
        x_ExpValue_vector[time_index]=real(sum(∫((𝛹ₓₜⁱ)'*xGridap*𝛹ₓₜⁱ)*dΩ))
    end
    return x_ExpValue_vector;
end

function position²_expectation_value(𝛹ₓₜ,Ω,dΩ,TrialSpace,x_component)
    x²Gridap=CellField(x->pow(x[x_component],2),Ω);
    x²_ExpValue_vector=zeros(Float64,length(𝛹ₓₜ));
    Threads.@threads for time_index in eachindex(𝛹ₓₜ)
        𝛹ₓₜⁱ=interpolate_everywhere(𝛹ₓₜ[time_index],TrialSpace)
        # ojo! tomamos la parte real porque se trata de la coord. espacial, pero puede ser complejo
        x²_ExpValue_vector[time_index]=real(sum(∫((𝛹ₓₜⁱ)'*x²Gridap*𝛹ₓₜⁱ)*dΩ))
    end
    return x²_ExpValue_vector;
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

@time begin
    println("Number of threads = ", Threads.nthreads());
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Resolvemos el problema 2D
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    existing_data=true
    # cantidad de FE y dominio espacial
    dom_2D=(-12.0*Angstrom_to_au,12.0*Angstrom_to_au,-4.9*Angstrom_to_au*γ,4.9*Angstrom_to_au*γ);
    # cantidad de FE por dimension (cantidad de intervalos)
    n_1D_r=5;n_1D_R=5;
    # tamaño del elemento 2D
    ΔrH=abs(dom_2D[2]-dom_2D[1])*(1.0/n_1D_r); ΔRH=abs(dom_2D[4]-dom_2D[3])*(1.0/n_1D_R);

    println("ΔrH=$(round(ΔrH/Angstrom_to_au,digits=2))[Å]; ΔRH=$(round(ΔRH/Angstrom_to_au,digits=2))[Å]; ΔχH=$(round(ΔRH/(Angstrom_to_au*γ),digits=2))[Å]")
    println("n_1D_r*n_1D_R=$(n_1D_r*n_1D_R) FE")

    # grilla de tamaño n²
    partition_2D=(n_1D_r,n_1D_R);
    # creamos modelo con elementos cartesianos
    model_2D=CartesianDiscreteModel(dom_2D,partition_2D);

    DOF_r,DOF_R,pts=space_coord_2D(dom_2D,ΔrH,ΔRH);

    # define boundary conditions (full dirichlet)
    dirichlet_values_2D=(0.0+im*0.0);
    dirichlet_tags_2D="boundary";

    Ω_2D,dΩ_2D,Γ_2D,dΓ_2D=measures(model_2D,3,dirichlet_tags_2D);
    reffe_2D=ReferenceFE(lagrangian,Float64,2);

    VH_2D=TestFESpace(model_2D,reffe_2D;vector_type=Vector{ComplexF64},conformity=:H1,dirichlet_tags=dirichlet_tags_2D);
    UH_2D=TrialFESpace(VH_2D,dirichlet_values_2D);

    R₁=-5.0*Angstrom_to_au;R₂=5.0*Angstrom_to_au;Rf=1.5*Angstrom_to_au;
    β=3.57*(1.0/(Angstrom_to_au*Angstrom_to_au));

    set_Rc_value=1; # set_Rc_value=1 or set_Rc_value=2

    if (set_Rc_value==1)
        Rc=1.5*Angstrom_to_au;  # screening parameter
        χ₀=-3.5*Angstrom_to_au; # Gaussian's center of init state
        n_eigenstate=1;         # fundamental state
    elseif (set_Rc_value==2) 
        Rc=5.0*Angstrom_to_au;
        χ₀=-1.5*Angstrom_to_au;
        n_eigenstate=2;  # first excited state
    end

    # Define bilinear forms and FE spaces
    pH_2D,qH_2D,rH_2D=eigenvalue_problem_functions((R₁,R₂,Rc,Rf);switch_potential = "Electron_Nuclear_Potential_2D")
    aH_2D,bH_2D=bilineal_forms(pH_2D,qH_2D,rH_2D,dΩ_2D);

    # solve eigenvalue problem
    nevH=5;
    probH_2D=EigenProblem(aH_2D,bH_2D,UH_2D,VH_2D;nev=nevH,tol=10^(-9),maxiter=1000,explicittransform=:none,sigma=-10.0);
    ϵH_2D,ϕH_2D=solve(probH_2D);

    # escribimos resultados en formato vtk
    println("Writing 2D problem eigenstates and eigenvalues")
    Threads.@threads for i in 1:nevH      
        writevtk(Ω_2D,path_images*"eigenprob_domr_2D_Rcvalue$(set_Rc_value)_grid$(n_1D_r)x$(n_1D_R)_num$(i)",cellfields=["ρr_eigenstates" => real((ϕH_2D[i])'*ϕH_2D[i])]);
    end

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Creamos la grilla 1D para resolver el problema electrónico
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # tipo de grilla
    grid_type="simple_line";
    # tamaño del elento 1D
    ΔrH_1D=ΔrH;
    ΔRH_1D=ΔRH;
    dom_1D_r=(dom_2D[1],dom_2D[2]);
    dom_1D_R=(dom_2D[3],dom_2D[4]);
    # (path,name,dom,MeshSize)
    par_1D_r=(path_models,grid_type*"_01_r_grid$(n_1D_r)x$(n_1D_R)",dom_1D_r,ΔrH_1D);
    par_1D_R=(path_models,grid_type*"_01_χ_grid$(n_1D_r)x$(n_1D_R)",dom_1D_R,ΔRH_1D);
    # creamos modelo
    model_1D_r=make_model(grid_type,par_1D_r);
    model_1D_R=make_model(grid_type,par_1D_R);
    # condiciones de contorno de tipo full dirichlet
    dirichlet_tags_1D=["left_point","right_point"];
    dirichlet_values_1D=[0.0+im*0.0,0.0+im*0.0];
    Ω_1D_r,dΩ_1D_r,Γ_1D_r,dΓ_1D_r=measures(model_1D_r,3,dirichlet_tags_1D);
    Ω_1D_R,dΩ_1D_R,Γ_1D_R,dΓ_1D_R=measures(model_1D_R,3,dirichlet_tags_1D);
    reffe_1D=reference_FEspaces(lagrangian,Float64,2);
    DOF_r_1D,pts_1D_r=space_coord_1D(dom_1D_r,ΔrH_1D);
    DOF_R_1D,pts_1D_R=space_coord_1D(dom_1D_R,ΔRH_1D);
    VH_1D_r=TestFESpace(model_1D_r,reffe_1D;vector_type=Vector{ComplexF64},conformity=:H1,dirichlet_tags=dirichlet_tags_1D);
    VH_1D_R=TestFESpace(model_1D_R,reffe_1D;vector_type=Vector{ComplexF64},conformity=:H1,dirichlet_tags=dirichlet_tags_1D);
    UH_1D_r=TrialFESpace(VH_1D_r,dirichlet_values_1D);
    UH_1D_R=TrialFESpace(VH_1D_R,dirichlet_values_1D);
    pH_1D_χ₀,qH_1D_χ₀,rH_1D_χ₀=eigenvalue_problem_functions((χ₀,R₁,R₂,Rc,Rf);switch_potential = "Electron_Nuclear_Potential_1D");
    aH_1D_χ₀,bH_1D_χ₀=bilineal_forms(pH_1D_χ₀,qH_1D_χ₀,rH_1D_χ₀,dΩ_1D_r);
    nevH_1D_χ₀=4;
    probH_1D_χ₀=EigenProblem(aH_1D_χ₀,bH_1D_χ₀,UH_1D_r,VH_1D_r;nev=nevH_1D_χ₀,tol=10^(-9),maxiter=1000,explicittransform=:none,sigma=-10.0);
    ϵH_1D_χ₀,ϕH_1D_χ₀=solve(probH_1D_χ₀);

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Creamos condición inicial
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    𝛹ₓ₀=create_initial_state_2D((χ₀,β,ϕH_1D_χ₀[n_eigenstate],Ω_2D,dΩ_2D,UH_2D);TypeOfFunction="OriginalFunctionBOAprox_v5");
    # escribimos resultados en archivo vtk
    println("Writing initial condition")
    writevtk(Ω_2D,path_images*"initial_condition__domrR_Rcvalue$(set_Rc_value)_grid$(n_1D_r)x$(n_1D_R)",cellfields=["ρₓ₀" => real((𝛹ₓ₀)'*𝛹ₓ₀)]);

    # chequeamos convergencia y escribimos resultados
    CheckConvergenceVector_χ=CheckConvergence(𝛹ₓ₀,ϕH_2D,UH_2D,dΩ_2D); # domino D={r,R}
    bin_outfile_name = path_images*"relative_error_convergence_study_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(CheckConvergenceVector_χ, bin_outfile_name; existing_file=existing_data);

    # tiempos adimensionales inicial y final
    t_start=0.0;t_end=200*Femtoseconds_to_au;
    Δt=100.0;   # time step
    n_points=round(Int,abs(t_end-t_start)*(1.0/Δt))+1;  # number of dicrete time points
    time_vec=[t_start+Δt*(i-1) for i in 1:n_points];
    println("Δt=$(Δt/Femtoseconds_to_au)[fs]; dim(time_vec)=$(length(time_vec))");

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Evolucionamos la función de onda y escribimos resultados
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    𝛹ₓₜ=evolution_schrodinger_v2(𝛹ₓ₀,ϕH_2D,ϵH_2D,UH_2D,dΩ_2D,time_vec); # domino D={r,R}

    println("Writing evolution of wave function")
    index_dat=0
    for i in 1:20:n_points
        global index_dat+=1
        writevtk(Ω_2D,path_images*"evolution_wave_function_domrR_Rcvalue$(set_Rc_value)_grid$(n_1D_r)x$(n_1D_R)_$(lpad(index_dat,3,'0'))",cellfields=["ρₓₜ" => real((𝛹ₓₜ[i])'*𝛹ₓₜ[i])]);
    end

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Calculamos las densidades de probabilidad reducidas y escribimos
    # resultados
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    electronic_ρ_matrix,nuclear_ρ_matrix=Partial_probability_density(𝛹ₓₜ,DOF_r,DOF_R,UH_2D,Ω_2D,dΩ_2D;TypeAproxDeltaFunction="StepFunction");

    println("Writing electronic probability density")
    electronic_ρ_matrix_plus_r=Matrix{Float64}(undef,length(electronic_ρ_matrix[:,1]),length(electronic_ρ_matrix[1,:])+1)
    electronic_ρ_matrix_plus_r[:,1]=DOF_r[:]
    electronic_ρ_matrix_plus_r[:,2:end]=electronic_ρ_matrix[:,:]
    bin_outfile_name = path_images*"electronic_density_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(electronic_ρ_matrix_plus_r ./ γ, bin_outfile_name; existing_file=existing_data);

    println("Writing nuclear probability density")
    nuclear_ρ_matrix_plus_R=Matrix{Float64}(undef,length(nuclear_ρ_matrix[:,1]),length(nuclear_ρ_matrix[1,:])+1)
    nuclear_ρ_matrix_plus_R[:,1]=DOF_R[:]
    nuclear_ρ_matrix_plus_R[:,2:end]=nuclear_ρ_matrix[:,:]
    bin_outfile_name = path_images*"nuclear_density_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(nuclear_ρ_matrix_plus_R, bin_outfile_name; existing_file=existing_data);

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Calculamos las entropías diferenciales de Shannon y
    # escribimos resultados. Dominio D={r,χ}
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    total_S_2D=TimeIndependet_Diff_Shannon_Entropy(𝛹ₓₜ,UH_2D,dΩ_2D)./ γ;

    # escribimos los resultados
    println("Writing total Shannon entropy")
    total_S_2D_plus_t=Matrix{Float64}(undef,length(total_S_2D[:,1]),2)
    total_S_2D_plus_t[:,1]=time_vec[:]
    total_S_2D_plus_t[:,2:end]=total_S_2D[:,:]
    bin_outfile_name = path_images*"total_shannon_entropy_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(total_S_2D_plus_t, bin_outfile_name; existing_file=existing_data);
    

    electronic_S=Reduced_TimeDependent_Diff_Shannon_Entropy(DOF_r,electronic_ρ_matrix) ./ γ .+ log(γ)
    println("Writing electronic Shannon entropy")
    electronic_S_plus_t=Matrix{Float64}(undef,length(electronic_S[:,1]),2)
    electronic_S_plus_t[:,1]=time_vec[:]
    electronic_S_plus_t[:,2:end]=electronic_S[:,:]
    bin_outfile_name = path_images*"electronic_shannon_entropy_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(electronic_S_plus_t, bin_outfile_name; existing_file=existing_data);

    nuclear_S=Reduced_TimeDependent_Diff_Shannon_Entropy(DOF_R,nuclear_ρ_matrix) ./ γ
    println("Writing nuclear Shannon entropy")
    nuclear_S_plus_t=Matrix{Float64}(undef,length(nuclear_S[:,1]),2)
    nuclear_S_plus_t[:,1]=time_vec[:]
    nuclear_S_plus_t[:,2:end]=nuclear_S[:,:]
    bin_outfile_name = path_images*"nuclear_shannon_entropy_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(nuclear_S_plus_t, bin_outfile_name; existing_file=existing_data);

    mutual_info=(electronic_S ./ γ .+ log(γ)) .+ (nuclear_S .- total_S_2D) ./ γ;
    println("Writing mutual information")
    mutual_info_plus_t=Matrix{Float64}(undef,length(mutual_info[:,1]),2)
    mutual_info_plus_t[:,1]=time_vec[:]
    mutual_info_plus_t[:,2:end]=mutual_info[:,:]
    bin_outfile_name = path_images*"mutual_information_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(mutual_info_plus_t, bin_outfile_name; existing_file=existing_data);

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Calculamos valores medios de la posición y varianza, y
    # escribimos resultados
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # dominio D={r,R}
    r_ExpValue=position_expectation_value(𝛹ₓₜ,Ω_2D,dΩ_2D,UH_2D,1) ./ γ ;
    println("Writing expectation value of electronic coordinate")
    r_ExpValue_plus_t=Matrix{Float64}(undef,length(r_ExpValue[:,1]),2)
    r_ExpValue_plus_t[:,1]=time_vec[:]
    r_ExpValue_plus_t[:,2:end]=r_ExpValue[:,:]
    bin_outfile_name = path_images*"ExpectationValue_r_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(r_ExpValue_plus_t ./ γ, bin_outfile_name; existing_file=existing_data);

    R_ExpValue=position_expectation_value(𝛹ₓₜ,Ω_2D,dΩ_2D,UH_2D,2) ./ (γ^2);
    println("Writing expectation value of nuclear coordinate")
    R_ExpValue_plus_t=Matrix{Float64}(undef,length(R_ExpValue[:,1]),2)
    R_ExpValue_plus_t[:,1]=time_vec[:]
    R_ExpValue_plus_t[:,2:end]=R_ExpValue[:,:]
    bin_outfile_name = path_images*"ExpectationValue_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(R_ExpValue_plus_t, bin_outfile_name; existing_file=existing_data);

    r²_ExpValue=position²_expectation_value(𝛹ₓₜ,Ω_2D,dΩ_2D,UH_2D,1) ./ (γ^2);
    R²_ExpValue=position²_expectation_value(𝛹ₓₜ,Ω_2D,dΩ_2D,UH_2D,2) ./ (γ^3);

    r_variance=sqrt.(r²_ExpValue.-(r_ExpValue.*r_ExpValue));
    println("Writing variance of electronic coordinate")
    r_variance_plus_t=Matrix{Float64}(undef,length(r_variance[:,1]),2)
    r_variance_plus_t[:,1]=time_vec[:]
    r_variance_plus_t[:,2:end]=r_variance[:,:]
    bin_outfile_name = path_images*"Variance_r_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(r_variance_plus_t, bin_outfile_name; existing_file=existing_data);

    R_variance=sqrt.(R²_ExpValue.-(R_ExpValue.*R_ExpValue));
    println("Writing variance of nuclear coordinate")
    R_variance_plus_t=Matrix{Float64}(undef,length(R_variance[:,1]),2)
    R_variance_plus_t[:,1]=time_vec[:]
    R_variance_plus_t[:,2:end]=R_variance[:,:]
    bin_outfile_name = path_images*"Variance_R_vs_time_Rc$(round(Rc/Angstrom_to_au;digits=2))_grid$(n_1D_r)x$(n_1D_R).bin"
    write_bin(R_variance_plus_t, bin_outfile_name; existing_file=existing_data);
end
