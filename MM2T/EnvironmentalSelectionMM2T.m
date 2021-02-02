function [SubPopulation,SubPop_Niching_value,SubPop_Scalarizing_value,FrontNo,CrowdDis] = EnvironmentalSelectionMM2T(Population,Pop_Niching_value,Pop_Scalarizing_value,N)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort([Pop_Niching_value Pop_Scalarizing_value],N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance([Pop_Niching_value Pop_Scalarizing_value],FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    SubPopulation = Population(Next);
    SubPop_Niching_value = Pop_Niching_value(Next);
    SubPop_Scalarizing_value = Pop_Scalarizing_value(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end