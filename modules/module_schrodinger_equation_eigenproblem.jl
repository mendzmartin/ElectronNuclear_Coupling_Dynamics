#!/usr/bin/julia

#=
    RUN COMMANDS
    Via REPL => julia
                include("module_schrodinger_equation_testing.jl")
    Via Bash => chmod +x module_schrodinger_equation_testing.jl
                ./module_schrodinger_equation_testing.jl
=#

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Definimos rutas a directorios específicos para buscar o guardar datos
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

path_models         = "../outputs/"*name_code*"/models/";
path_images         = "../outputs/"*name_code*"/images/";
path_modules        = "../modules/"
path_gridap_makie   = "../gridap_makie/";
path_videos         = "./videos/";
path_plots          = "../outputs/"*name_code*"/plots/";


#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Activamos proyecto e intalamos paquetes para FEM
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

# activamos el proyecto "gridap_makie" donde se intalarán todos los paquetes
import Pkg; Pkg.activate(path_gridap_makie);

install_packages=false;
if install_packages
    import Pkg
    Pkg.add("Gridap");
    Pkg.add("GridapGmsh");
    Pkg.add("Gmsh");
    Pkg.add("FileIO");
end

using Gridap;
using GridapGmsh;
using Gmsh;
using Gridap.CellData;  # para construir condición inicial interpolando una función conocida
using Gridap.FESpaces;  # para crear matrices afines a partir de formas bilineales
using Gridap.Algebra;   # para utilizar operaciones algebraicas con Gridap

install_packages=false;
if install_packages
    import Pkg
    Pkg.add("Plots")
end
using Plots;

# crear directorios en caso de no haberlo hecho
create_directories = false;
if (create_directories==true)
    mkdir(path_models);
    mkdir(path_images);
    mkdir(path_plots);
end

using FileIO;

# en caso de querer plotear dentro de Jupiter Notebook
#  debemos usar algunos paquetes. (no funciona en VSCode)
plot_s = false;
if plot_s
    using GridapMakie, GLMakie; # Para graficar 
    using FileIO;               # Gráficos y salidas
end

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Instalamos otros paquetes útiles
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

using Printf; # para imprimir salidas con formatos

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Instalamos paquetes para operaciones algebraicas
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

install_packages=false;
if install_packages
    import Pkg;
    Pkg.add("LinearAlgebra");
    Pkg.add("SparseArrays");
    Pkg.add("LinearAlgebra");
    Pkg.add("Arpack");
end
using LinearAlgebra;
using SparseArrays;
using SuiteSparse;
using Arpack;

install_packages=false;
if install_packages
    import Pkg;
    Pkg.add("DataInterpolations");
    Pkg.add("BenchmarkTools");
end
using DataInterpolations;   # interpolation function package (https://github.com/PumasAI/DataInterpolations.jl)
using BenchmarkTools;       # benchmarks and performance package (https://juliaci.github.io/BenchmarkTools.jl/stable/)

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Importamos módulos
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

include(path_modules*"module_eigen.jl");            # módulo para resolver problema de autovalores
include(path_modules*"module_mesh_generator.jl");   # módulo para construir grilla (1D)

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Seteo de variables globales
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

# declaramos parámetros constantes (utilizando sistema atómico de unidades)
const m=1.0;            # electron mass
const M=2000.0*m;       # proton mass
const ħ=1.0;            # Planck constant
const γ=sqrt(M*(1.0/m));    # scaling factor (R=γχ)

# set unit convertion constant
const Bohr_radius_meter=5.29177210903e−11;                        # [m]
const Angstrom_to_meter=1e−10;                                    # [m/Å]
const Angstrom_to_au=Angstrom_to_meter*(1.0/Bohr_radius_meter);   # [au/Å]
const Femtoseconds_to_au=(1.0/0.0218884);                         # [au/fs]

α=im*ħ*0.5*(1.0/m);                  # factor multiplicativo energía cinética
αconst(ω)=-im*0.5*m*(ω*ω)*(1.0/ħ);   # factor multiplicativo potencial armónico

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Funciones útiles
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

# the triangulation and integration aproximated Lebesgue measure
function measures(model,degree,tags_boundary)
    # triangulation of the integration domain
    Ω=Triangulation(model);
    dΩ=Measure(Ω,degree);
    # triangulation of the boundary domain whit boundary conditions
    Γ=BoundaryTriangulation(model,tags=tags_boundary);
    dΓ=Measure(Γ,degree)
    return Ω,dΩ,Γ,dΓ;
end
# definimos espacios de referencia
function reference_FEspaces(method,type,order)
    reff=ReferenceFE(method,type,order);
    return reff;
end

# funciones para problema de autovalores (ecuaciones de Sturm Liouville)

function eigenvalue_problem_functions(params;switch_potential = "QHO_1D")
    if (switch_potential == "QHO_1D")
        # caso de potencial tipo quantum harmonic oscillator 1D (QHO)
        @printf("Set quantum harmonic oscillator 1D potential\n");
        ω,x₁=params;
        pₕ_QHO_1D(x) = 0.5*(ħ*ħ)*(1.0/m);                                      # factor para energía cinética
        qₕ_QHO_1D(x) = 0.5*m*(ω*ω)*(x[1]-x₁)*(x[1]-x₁);                        # oscilador armónico 1D centrado en x₁
        rₕ_QHO_1D(x) = 1.0;
        return pₕ_QHO_1D,qₕ_QHO_1D,rₕ_QHO_1D;
    elseif (switch_potential == "QHO_2D")
        # caso de potencial tipo quantum harmonic oscillator 2D (QHO)
        @printf("Set quantum harmonic oscillator 2D potential\n");
        ω,x₁,y₁=params;
        pₕ_QHO_2D(x) = 0.5*(ħ*ħ)*(1.0/m);                                       # factor para energía cinética
        qₕ_QHO_2D(x) = 0.5*m*(ω*ω)*((x[1]-x₁)*(x[1]-x₁)+(x[2]-y₁)*(x[2]-y₁));   # oscilador armónico 2D centrado en (x₁,y₁)
        rₕ_QHO_2D(x) = 1.0;
        return pₕ_QHO_2D,qₕ_QHO_2D,rₕ_QHO_2D;
    elseif (switch_potential == "FWP")
        # caso de potencial tipo finite well potential (FWP)
        @printf("Set quantum finite well potential\n");
        V₀_FWP,a_FWP=params;
        pₕ_FWP(x) = 0.5*(ħ*ħ)*(1.0/m);                                          # factor para energía cinética
        qₕ_FWP(x) = interval.(x[1],-a_FWP,a_FWP,V₀_FWP)
        rₕ_FWP(x) = 1.0;
        return pₕ_FWP,qₕ_FWP,rₕ_FWP;
    elseif (switch_potential == "Electron_Nuclear_Potential_1D")
        # caso de potencial tipo interacción electron-nucleo en pozo nuclear
        # @printf("Set Electron-Nuclear potential with fixed R\n");
        R,R₁,R₂,Rc,Rf=params;
        pₕ_ENP_1D(x) = 0.5*(ħ*ħ)*(1.0/m);                                          # factor para energía cinética
        qₕ_ENP_1D(x) = CoulombPotential(R,R₁)+CoulombPotential(R,R₂)+
            Aprox_Coulomb_Potential(x[1],R₁,Rf)+Aprox_Coulomb_Potential(x[1],R,Rc)+Aprox_Coulomb_Potential(x[1],R₂,Rf)
        rₕ_ENP_1D(x) = 1.0;
        return pₕ_ENP_1D,qₕ_ENP_1D,rₕ_ENP_1D;
    elseif (switch_potential == "Electron_Nuclear_Potential_2D")
        # caso de potencial tipo interacción electron-nucleo en pozo nuclear
        @printf("Set Electron-Nuclear potential\n");
        R₁,R₂,Rc,Rf=params;
        pₕ_ENP_2D(x) = 0.5*(ħ*ħ)*(1.0/m);     # factor para energía cinética
        qₕ_ENP_2D(x) = CoulombPotential(x[2]*(1.0/γ),R₁)+CoulombPotential(x[2]*(1.0/γ),R₂)+
            Aprox_Coulomb_Potential(x[1],R₁,Rf)+Aprox_Coulomb_Potential(x[1],x[2]*(1.0/γ),Rc)+Aprox_Coulomb_Potential(x[1],R₂,Rf)
        rₕ_ENP_2D(x) = 1.0;
        return pₕ_ENP_2D,qₕ_ENP_2D,rₕ_ENP_2D;
    end
end

# Formas bilineales para problema de autovalores
function bilineal_forms(p,q,r,dΩ)
    a(u,v) = ∫(p*(∇(v)⋅∇(u))+q*v*u)*dΩ;
    b(u,v) = ∫(r*u*v)*dΩ;
    return a,b;
end

# Formas bilineales para problema de autovalores (parte Re e Im por separado)

function bilineal_forms_ReImParts(p,q,r,dΩ)
    a₁((u₁,v₁))=∫(p*(∇(v₁)⋅∇(u₁))+q*(v₁*u₁))*dΩ;
    b₁((u₁,v₁))=∫(r*(v₁*u₁))*dΩ;

    a₂((u₂,v₂))=∫(p*(∇(v₂)⋅∇(u₂))+q*(v₂*u₂))*dΩ;
    b₂((u₂,v₂))=∫(r*(v₂*u₂))*dΩ;

    a((u₁,u₂),(v₁,v₂)) = a₁((u₁,v₁))+a₂((u₂,v₂))
    b((u₁,u₂),(v₁,v₂)) = b₁((u₁,v₁))+b₂((u₂,v₂))
    return a,b;
end

# Norma L₂
function norm_L2(u,dΩ)
    return sqrt(real(sum(∫(u'*u)*dΩ)));
end

# funciones para hamiltoniano 2x2 1D
α₁(x,(x₁,x₂,ω))=αconst(ω)*(x[1]-x₁)*(x[1]-x₁); # oscilador armónico 1D centrado en x₁
α₂(x,(x₁,x₂,ω))=αconst(ω)*(x[1]-x₂)*(x[1]-x₂); # oscilador armónico 1D centrado en x₂

#=
    función para obtener los puntos discretos de la grilla (valuados)
    y un vector pts que almacena dichos puntos
=#
function space_coord_1D(dom,Δx)
    nx=round(Int,abs(dom[2]-dom[1])/Δx)+1; # cantidad de puntos en dirección x
    x=[dom[1]+Δx*(i-1) for i in 1:nx];
    pts=[Point(x[i]) for i in 1:nx];
    return x,pts;
end

function space_coord_2D(dom,Δx,Δy)
    nx=round(Int,abs(dom[2]-dom[1])/Δx)+1; # cantidad de puntos en dirección x
    ny=round(Int,abs(dom[4]-dom[3])/Δy)+1; # cantidad de puntos en dirección y
    x=[dom[1]+Δx*(i-1) for i in 1:nx];
    y=[dom[3]+Δy*(i-1) for i in 1:ny];
    pts=[Point(x[i],y[j]) for i in 1:nx for j in 1:ny];
    return x,y,pts;
end

#=
    función para calcular normalización de autoestados de
    un hamiltoniano 1D
=#
function normalization_eigenstates(ϕ,TrialSpace,dΩ)
    nom_vec=zeros(Float64,length(ϕ))
    for i in 1:length(ϕ)
        ϕᵢ=interpolate_everywhere(ϕ[i],TrialSpace);
        nom_vec[i]=norm_L2(ϕ[i],dΩ)
    end
    return nom_vec;
end
#=
    función para calcular normalización de autoestados de
    un hamiltoniano 2D
=#
function normalization_eigenstates_multifield(ϕ,TrialSpace,dΩ)
    nom_vec₁₂=zeros(Float64,length(ϕ))
    for i in 1:length(ϕ)
        ϕᵢ=interpolate_everywhere(ϕ[i],TrialSpace);
        ϕ¹ᵢ,ϕ²ᵢ=ϕᵢ
        norm_ϕ¹ᵢ=norm_L2(ϕ¹ᵢ,dΩ)
        norm_ϕ²ᵢ=norm_L2(ϕ²ᵢ,dΩ)
        nom_vec₁₂[i]=norm_ϕ¹ᵢ+norm_ϕ²ᵢ
    end
    return nom_vec₁₂;
end
#=
    función para chequear ortogonalidad de autoestados de
    un hamiltoniano 2D
=#
function OrthoCheck_multifield(ϕ,TrialSpace,dΩ)
    nev=length(ϕ)
    OrthoVector=zeros(Float64,nev^2-nev);
    index=1
    for i in 1:nev
        ϕᵢ=interpolate_everywhere(ϕ[i],TrialSpace);
        ϕ¹ᵢ,ϕ²ᵢ=ϕᵢ
        for j in 1:nev
            if (i ≠ j)
                ϕⱼ=interpolate_everywhere(ϕ[j],TrialSpace);
                ϕ¹ⱼ,ϕ²ⱼ=ϕⱼ
                OrthoVector[index]=abs(sum(∫(ϕ¹ⱼ'*ϕ¹ᵢ)*dΩ)+sum(∫(ϕ²ⱼ'*ϕ²ᵢ)*dΩ))
                index+=1
            end
        end
    end
    return OrthoVector;
end

function OrthoCheck(ϕ,TrialSpace,dΩ)
    nev=length(ϕ)
    OrthoVector=zeros(Float64,nev^2-nev);
    index=1
    for i in 1:nev
        ϕᵢ=interpolate_everywhere(ϕ[i],TrialSpace);
        for j in 1:nev
            if (i ≠ j)
                ϕⱼ=interpolate_everywhere(ϕ[j],TrialSpace);
                OrthoVector[index]=abs(sum(∫(ϕⱼ'*ϕᵢ)*dΩ))
                index+=1
            end
        end
    end
    return OrthoVector;
end

function OrthoCheck_v2(ϕ,TrialSpace,dΩ)
    nev=length(ϕ)
    OrthoVector=zeros(Float64,round(Int,(nev^2-nev)/2));
    index=1
    for i in 2:nev
        ϕᵢ=interpolate_everywhere(ϕ[i],TrialSpace);
        for j in 1:(i-1)
            ϕⱼ=interpolate_everywhere(ϕ[j],TrialSpace);
            OrthoVector[index]=abs(sum(∫(ϕⱼ'*ϕᵢ)*dΩ))
            index+=1
        end
    end
    return OrthoVector;
end

#=
    función para calcular la populación de estados
=#
function Populations_multifield(𝛹ₓₜ,TrialSpace,dΩ)
    dimₜ=length(𝛹ₓₜ)
    p¹ₜ=zeros(Float64,dimₜ);
    p²ₜ=zeros(Float64,dimₜ);

    for i in 1:dimₜ
        𝛹ₓₜᵢ=interpolate_everywhere(𝛹ₓₜ[i],TrialSpace);
        𝛹¹ₓₜᵢ,𝛹²ₓₜᵢ=𝛹ₓₜᵢ
        norm_𝛹¹ₓₜᵢ=norm_L2(𝛹¹ₓₜᵢ,dΩ)
        norm_𝛹²ₓₜᵢ=norm_L2(𝛹²ₓₜᵢ,dΩ)
        p¹ₜ[i]=real(sum(∫(𝛹¹ₓₜᵢ'*𝛹¹ₓₜᵢ)*dΩ))/(norm_𝛹¹ₓₜᵢ)
        p²ₜ[i]=real(sum(∫(𝛹²ₓₜᵢ'*𝛹²ₓₜᵢ)*dΩ))/(norm_𝛹²ₓₜᵢ)
    end

    return p¹ₜ,p²ₜ;
end

#=
    function to calculate differential Shannon entropy
=#
"""
    https://en.wikipedia.org/wiki/Natural_logarithm
"""
function ln_aprox(x,n)
    result = 1.0
    for i in 1:n
        result = pow(-1.0,i-1)*pow((x-1),i)*(1.0/i)
    end
    return result
end

function pow(x,n)
    result = 1.0
    for i in 1:n
        result=result*x
    end
    return result
end

function TimeIndependet_Diff_Shannon_Entropy(𝛹ₓ,TrialSpace,dΩ)
    dim𝛹ₓ=length(𝛹ₓ)
    S=zeros(Float64,dim𝛹ₓ)
    for i in 1:dim𝛹ₓ
        𝛹ₓᵢ=interpolate_everywhere(𝛹ₓ[i],TrialSpace);
        𝛹ₓᵢ=𝛹ₓᵢ/norm_L2(𝛹ₓᵢ,dΩ);
        ρₓᵢ=real(𝛹ₓᵢ'*𝛹ₓᵢ)
        (sum(integrate(ρₓᵢ,dΩ))==0.0) ? (S[i]=0.0) : (S[i]=-sum(integrate(ρₓᵢ*(log∘ρₓᵢ),dΩ)))
    end
    return S;
end

#=
    funcion auxiliar para calcular función de heaviside
    y construir un pozo cuadrado de potencial
=#

function heaviside(x)
    0.5*(sign(x)+1)
 end

function interval(x,x₁,x₂,A)
   A*(heaviside(x-x₁)-heaviside(x-x₂))
end

function AproxDiracDeltaFunction(x,params;TypeFunction="StepFunction")
    if (TypeFunction=="BumpFunction")
        # https://en.wikipedia.org/wiki/Dirac_delta_function
        # https://en.wikipedia.org/wiki/Bump_function
        x₀,δnorm,component=params
        a=10000;b=1.0;
        δ=(1.0/(abs(b)*sqrt(π)))*exp(-a*pow((x[component]-x₀)*(1.0/b),2))*(1.0/δnorm)
    elseif (TypeFunction=="StepFunction")
        x₀,δnorm,component,Δx=params
        (abs(x[component]-x₀)≤(0.5*Δx)) ? δ=(1.0/Δx)*(1.0/δnorm) : δ=0.0
    end
    return δ
end

CoulombPotential(r,r₀)=1.0/abs(r₀-r);

install_packages=false;
if install_packages
    Pkg.add("SpecialFunctions"); # https://specialfunctions.juliamath.org/stable/
end

using SpecialFunctions;
Aprox_Coulomb_Potential(r,r₀,R)=-erf(abs(r₀-r)*(1.0/R))*CoulombPotential(r,r₀)


#=
    Function to find initial state descomposition coefficients
        when base functions are not orthogonal each other
=#
function CoeffInit_no_orthogonal(𝛹ₓ₀,ϕₙ,TrialSpace,dΩ)
    dim=length(ϕₙ)
    InnerProdEigenvecs=zeros(ComplexF64,dim,dim);   # matriz global de inversas de productos internos entre autoestados
    InnerProdBC=zeros(ComplexF64,dim);              # vector global de productos internos entre autoestados y estado inicial
    # primer submatriz n✖n y subvector n✖1
    for i in 1:dim
        ϕᵢ=interpolate_everywhere(ϕₙ[i],TrialSpace);
        InnerProdBC[i]=sum(∫(ϕᵢ'*𝛹ₓ₀)*dΩ)
        for j in 1:i
            ϕⱼ=interpolate_everywhere(ϕₙ[j],TrialSpace);
            InnerProdEigenvecs[i,j]=sum(∫(ϕᵢ'*ϕⱼ)*dΩ)
            if (i≠j) # optimización por simetría
                InnerProdEigenvecs[j,i]=conj(InnerProdEigenvecs[i,j])
            end
        end
    end
    # x=A\b
    coeffvec₁₂=InnerProdEigenvecs\InnerProdBC;
    return coeffvec₁₂;
end

#=
    Function to find initial state descomposition coefficients
        when base functions are orthogonal each other
=#
function CoeffInit(𝛹ₓ₀,ϕₙ,TrialSpace,dΩ)
    dim=length(ϕₙ)
    coeffvec₁₂=zeros(ComplexF64,dim); # vector global de productos internos entre autoestados y estado inicial
    for i in 1:dim
        ϕᵢ=interpolate_everywhere(ϕₙ[i],TrialSpace);
        coeffvec₁₂[i]=sum(∫(ϕᵢ'*𝛹ₓ₀)*dΩ)
    end
    return coeffvec₁₂;
end

function CheckConvergence(𝛹ₓ₀,ϕₙ,TrialSpace,dΩ)
    coeffvec₁₂=CoeffInit(𝛹ₓ₀,ϕₙ,TrialSpace,dΩ)
    sum_coeff=zeros(Float64,length(ϕₙ));
    sum_coeff[1]=real((coeffvec₁₂[1])'*coeffvec₁₂[1])
    for i in 2:length(ϕₙ)
        sum_coeff[i]=sum_coeff[i-1]+real((coeffvec₁₂[i])'*coeffvec₁₂[i])
    end
    return sum_coeff;
end

#=
    Function to evolve quantum system
=#
function evolution_schrodinger(𝛹ₓ₀,ϕₙ,ϵₙ,TrialSpace,dΩ,time_vec)
    dim_time=length(time_vec)
    # calculamos los coeficientes de la superposición lineal
    coeffvec₁₂=CoeffInit(𝛹ₓ₀,ϕₙ,TrialSpace,dΩ)
    𝛹ₓₜ=Vector{CellField}(undef,dim_time);
    # inicializamos en cero el vector de onda
    ϕ₁=interpolate_everywhere(ϕₙ[1],TrialSpace);
    for i in 1:dim_time
        𝛹ₓₜ[i]=interpolate_everywhere(0.0*ϕ₁,TrialSpace)
    end
    for i in 1:dim_time
        for j in 1:length(ϵₙ)
            𝛹ₓₜⁱ=interpolate_everywhere(𝛹ₓₜ[i],TrialSpace)
            ϕⱼ=interpolate_everywhere(ϕₙ[j],TrialSpace);
            factor=coeffvec₁₂[j]*exp(-im*(1.0/ħ)*real(ϵₙ[j])*time_vec[i])
            𝛹ₓₜ[i]=interpolate_everywhere((𝛹ₓₜⁱ+factor*ϕⱼ),TrialSpace)
        end
        # normalizamos la función de onda luego de cada evolución
        norm_switch=true
        if norm_switch
            𝛹ₓₜⁱ=interpolate_everywhere(𝛹ₓₜ[i],TrialSpace);
            Norm𝛹ₓₜⁱ=norm_L2(𝛹ₓₜ[i],dΩ)
            𝛹ₓₜ[i]=interpolate_everywhere((𝛹ₓₜⁱ*(1.0/Norm𝛹ₓₜⁱ)),TrialSpace)
        end
        # recalculamos los coeficientes de la superposición lineal
        coeffvec₁₂=CoeffInit(𝛹ₓₜ[i],ϕₙ,TrialSpace,dΩ)
        println("run step = $(i)/$(dim_time)");
    end
    return 𝛹ₓₜ;
end

function evolution_schrodinger_v2(𝛹ₓ₀,ϕₙ,ϵₙ,TrialSpace,dΩ,time_vec)
    dim_time=length(time_vec)
    # calculamos los coeficientes de la superposición lineal
    coeffvec₁₂=CoeffInit(𝛹ₓ₀,ϕₙ,TrialSpace,dΩ)
    𝛹ₓₜ=Vector{CellField}(undef,dim_time);
    # inicializamos en cero el vector de onda
    ϕ₁=interpolate_everywhere(ϕₙ[1],TrialSpace);
    for i in 1:dim_time
        𝛹ₓₜ[i]=interpolate_everywhere(0.0*ϕ₁,TrialSpace)
    end
    for i in 1:dim_time
        for j in 1:length(ϵₙ)
            𝛹ₓₜⁱ=interpolate_everywhere(𝛹ₓₜ[i],TrialSpace)
            ϕⱼ=interpolate_everywhere(ϕₙ[j],TrialSpace);
            factor=coeffvec₁₂[j]*exp(-im*(1.0/ħ)*real(ϵₙ[j])*time_vec[i])
            𝛹ₓₜ[i]=interpolate_everywhere((𝛹ₓₜⁱ+factor*ϕⱼ),TrialSpace)
        end
        # normalizamos la función de onda luego de cada evolución
        norm_switch=true
        if norm_switch
            𝛹ₓₜⁱ=interpolate_everywhere(𝛹ₓₜ[i],TrialSpace);
            Norm𝛹ₓₜⁱ=norm_L2(𝛹ₓₜ[i],dΩ)
            𝛹ₓₜ[i]=interpolate_everywhere((𝛹ₓₜⁱ*(1.0/Norm𝛹ₓₜⁱ)),TrialSpace)
        end
    end
    return 𝛹ₓₜ;
end