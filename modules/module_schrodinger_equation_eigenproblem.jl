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
    import Pkg
    Pkg.add("LinearAlgebra");
    Pkg.add("SparseArrays");
    Pkg.add("LinearAlgebra");
    Pkg.add("Arpack");
end
using LinearAlgebra;
using SparseArrays;
using SuiteSparse;
using Arpack;

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Importamos módulos
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

include(path_modules*"module_eigen.jl");            # módulo para resolver problema de autovalores
include(path_modules*"module_mesh_generator.jl");   # módulo para construir grilla (1D)

#= +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++ Seteo de variables globales
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ =#

# declaramos parámetros constantes (utilizando sistema atómico de unidades)
const m=1.0;        # electron mass
const M=200.0*m;    # proton mass
const ħ=1.0;        # Planck constant

α=im*ħ*0.5*(1.0/m);               # factor multiplicativo energía cinética
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
    elseif (switch_potential == "Electron_Nuclear_Potential")
        # caso de potencial tipo interacción electron-nucleo en pozo nuclear
        @printf("Set Electron-Nuclear potential\n");
        R,R₁,R₂,Rc,Rf=params;
        pₕ_ENP(x) = 0.5*(ħ*ħ)*(1.0/m+1.0/M);                                          # factor para energía cinética
        qₕ_ENP(x) = CoulombPotential(R,R₁)+CoulombPotential(R,R₂)+
            Aprox_Coulomb_Potential(x[1],R₁,Rf)+Aprox_Coulomb_Potential(x[1],R,Rc)+Aprox_Coulomb_Potential(x[1],R₂,Rf)
        rₕ_ENP(x) = 1.0;
        return pₕ_ENP,qₕ_ENP,rₕ_ENP;
    end
end

# Formas bilineales para problema de autovalores (parte Re e Im por separado)

function bilineal_forms_ReImParts(p,q,r,dΩ;switch_potential="QHO_1D")
    a₁((u₁,v₁))=∫(p*(∇(v₁)⋅∇(u₁))+q*(v₁*u₁))dΩ;
    b₁((u₁,v₁))=∫(r*(v₁*u₁))dΩ;

    a₂((u₂,v₂))=∫(p*(∇(v₂)⋅∇(u₂))+q*(v₂*u₂))dΩ;
    b₂((u₂,v₂))=∫(r*(v₂*u₂))dΩ;

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
function space_coord(dom,Δx)
    x=[dom[1]+abs(dom[2]-dom[1])*Δx*i for i in 1:convert(Int,1.0/Δx)];
    pts=[Point(x[i]) for i in 1:convert(Int,1.0/Δx)];
    return x,pts;
end

#=
    función para calcular normalización de autoestados de
    un hamiltoniano 1D
=#
function normalization_eigenstates_1D(ϕ,TrialSpace,dΩ)
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
function normalization_eigenstates_2D(ϕ,TrialSpace,dΩ)
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
function OrthoCheck_2D(ϕ,TrialSpace,dΩ)
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
#=
    función para calcular la populación de estados
=#
function Populations_2D(𝛹ₓₜ,TrialSpace,dΩ)
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
function Diff_Shannon_Entropy_1D(𝛹ₓ,TrialSpace,dΩ,pts)
    dim𝛹ₓ=length(𝛹ₓ)
    𝛹ₓᵢ=interpolate_everywhere(𝛹ₓ[1],TrialSpace);
    S=zeros(Float64,dim𝛹ₓ)
    for i in 1:dim𝛹ₓ
        𝛹ₓᵢ=interpolate_everywhere(𝛹ₓ[i],TrialSpace);
        𝛹ₓᵢ=𝛹ₓᵢ/norm_L2(𝛹ₓᵢ,dΩ);

        ρₓᵢ=𝛹ₓᵢ'*𝛹ₓᵢ
        S[i]=real(sum(∫(ρₓᵢ->(-ρₓᵢ*log(ρₓᵢ)))*dΩ))

        # ρₓᵢ=(𝛹ₓᵢ'.(pts)).*(𝛹ₓᵢ.(pts));
        # S[i]=real(sum(∫(ρₓᵢ*log.(ρₓᵢ))*dΩ));
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

CoulombPotential(r,r₀)=1.0/abs(r₀-r);

install_packages=false;
if install_packages
    Pkg.add("SpecialFunctions"); # https://specialfunctions.juliamath.org/stable/
end

using SpecialFunctions;
Aprox_Coulomb_Potential(r,r₀,R)=-erf(abs(r₀-r)*(1.0/R))*CoulombPotential(r,r₀)