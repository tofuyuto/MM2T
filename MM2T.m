function MM2T(Global)
% <algorithm> <EMMO>
% Multi-objective Mult-modal to Two-objective 
% type --- 3 --- The type of aggregation function
% off --- 1 --- The type of offspring generation (1 = 'SBX') , (2 = 'DE')
% S --- 10 --- Subpopulation size

%------------------------------- Reference --------------------------------
% Takafumi Fukase, Naoki Masuyama, Yuusuke Nojima, Hisao Ishibuchi, 
% A Constrained Multi objective Evolutionary Algorithm Based on 
% Transformation to Two objective Optimization Problems, In Proc. of
% FAN2019, Toyama, 2019.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [type,off,S,seed] = Global.ParameterSet(3,1,10,Global.seed);
    rng('default');
    rng(seed);
    % 初期個体合わせると+1
    saveOtherOutput = 10;
    S = floor(Global.N / S);
    % parameter for post-processing method
    outputPrimary = false;
    outputTeritary = false;
    reducedSize = Global.N;
    
    %% Generate the weight vectors
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    
    %% Generate random population
    Population = Global.Initialization();
    
    Nadir = max(Population.objs,[],1);
    Z = min(Population.objs,[],1);
    
    %% 初期個体を Global.N 個の初期サブ個体群に収容
    UpdatePopulation = Population;
    minusNich = decide_nich(Population.decs,Global);
    for i = 1 : Global.N
        Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Population.objs,Z,Nadir);
        
        % スカラー化関数と制約違反量を目的関数値とする個体群を作成．
        [SubPopulation(i,:),SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:),FrontNo(i,:),CrowdDis(i,:)] = EnvironmentalSelectionMM2T(UpdatePopulation,minusNich,Scalarizing_value,S);
    end

    Offspring = Population;
    
    % Output to otherOutputs Folder.
    if Global.save > 0
        OutputWeightedVectorsInfo(Global,W);
        OutputSubPopulationInfo(Global,SubPopulation,SubPop_Niching_value,SubPop_Scalarizing_value);
    end
    
    %% Optimization
    while Global.NotTermination(Population)
        Nadir = Z;
        % サブ個体群内で子個体の生成と参照点の更新．
        Offspring_gennum = 1;
        I = randperm(Global.N);
        for i = 1 : Global.N
            % Generate an offspring
            if off == 1
                MatingPool = TournamentSelection(2,2,SubPop_Scalarizing_value(I(i),:),SubPop_Niching_value(I(i),:));
                Offspring(Offspring_gennum) = GAhalf(SubPopulation(I(i),MatingPool));
            elseif off == 2
                MatingPool = TournamentSelection(2,3,SubPop_Scalarizing_value(I(i),:),SubPop_Niching_value(I(i),:));
                Offspring(Offspring_gennum) = DE(SubPopulation(I(i),MatingPool(1)),SubPopulation(I(i),MatingPool(2)),SubPopulation(I(i),MatingPool(3)));
            end
            Nadir = max(Nadir,Offspring(Offspring_gennum).obj); 
            Z = min(Z,Offspring(Offspring_gennum).obj);
            Offspring_gennum = Offspring_gennum + 1;
        end
        
        % 現在の現個体群と全ての子個体群を合わせた Gloal.N + S 個でサブ個体群の更新を行う．
        Offspring_Objs = Offspring.objs;       
        for i = 1 : Global.N
            % 子個体に目的関数（スカラー化関数と制約違反量）をセット．
            Curent_Objs = SubPopulation(i,:).objs;
            Offspring_Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Offspring_Objs,Z,Nadir);
            Curent_Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Curent_Objs,Z,Nadir);
            
            Update_minusNiching = decide_nich([SubPopulation(i,:).decs; Offspring.decs],Global);
            Update_Scalarizing_value = [Curent_Scalarizing_value; Offspring_Scalarizing_value];
            
            % Gloal.N + S 個でサブ個体群 i の更新．
            [SubPopulation(i,:),SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:),FrontNo(i,:),CrowdDis(i,:)] = EnvironmentalSelectionMM2T([SubPopulation(i,:),Offspring],Update_minusNiching,Update_Scalarizing_value,S);
        end
        
        % Output to otherOutputs Folder.
        if Global.save > 0
            if rem(saveOtherOutput*Global.evaluated, Global.evaluation) == 0
                OutputSubPopulationInfo(Global,SubPopulation,SubPop_Niching_value,SubPop_Scalarizing_value);
            end
        end
        
        if ~outputPrimary && ~outputTeritary
            % サブ個体群を1つの個体群に集約．
            evaluated = Global.GetObj().evaluated;
            if Global.evaluated < Global.evaluation
                for i = 1 : Global.N
                    %{
                    Population(i) = SubPopulation(i,TournamentSelection(S,1,SubPop_Scalarizing_value(i,:),SubPop_Niching_value(i,:)));
                    Global.change_evaluated(evaluated);
                    %}
                    Population(S*(i-1)+1:S*i) = SubPopulation(i,:);
                    Global.change_evaluated(evaluated);
                end
            else
                % Output
                for i = 1 : Global.N
                    %{
                    Population(i) = SubPopulation(i,TournamentSelection(S,1,SubPop_Scalarizing_value(i,:),SubPop_Niching_value(i,:)));
                    Global.change_evaluated(evaluated);
                    %}
                    Population(S*(i-1)+1:S*i) = SubPopulation(i,:);
                    Global.change_evaluated(evaluated);
                end
            end
        else
            % Post-processing method for selecting solutions
            Feasible     = find(all(SubPopulation.cons<=0,2));
            NonDominated = NDSort(SubPopulation(Feasible).objs,1) == 1;
            Feasible     = reshape(Feasible,[Global.N S]);
            NonDominated = reshape(NonDominated, [Global.N S]);
            Population = reshape(SubPopulation(Feasible(NonDominated)), 1, []);
            beforeN = Global.N;
            Global.N = size(Population,2);

            % primary selection
            if outputPrimary
                primaryPopulation = INDIVIDUAL.empty;
                for i = 1 : size(SubPopulation,1)
                    PopinTergetSubProb = SubPopulation(i,NonDominated(i,:));
                    ScalarizingFuncValue = SubPop_Scalarizing_value(i,NonDominated(i,:));
                    if ~isempty(ScalarizingFuncValue)
                        [~,minIndex] = min(ScalarizingFuncValue,[],2);
                        primaryPopulation(end+1) = PopinTergetSubProb(minIndex);
                    end
                end
                Population = primaryPopulation;
            end

            % reduced selecting
            if outputTeritary
                if Global.N <= reducedSize
                    reducedPopulation = Population;
                else
                    reducedPopulation = INDIVIDUAL.empty;
                    NormalizedPopDecs = (Population.decs - Global.lower)./(Global.upper - Global.lower);
                    isSelected = false(1,Global.N);
                    PopDecDistance = pdist2(NormalizedPopDecs,NormalizedPopDecs,'euclidean');

                    firstPopID = randperm(Global.N,1);
                    isSelected(firstPopID) = true;
                    reducedPopulation(end+1) = Population(firstPopID);
                    minDistance = PopDecDistance(firstPopID,:);

                    while size(reducedPopulation,2) ~= reducedSize
                        [~,TergetPopID] = max(minDistance.*~isSelected,[],2);
                        reducedPopulation(end+1) = Population(TergetPopID);
                        isSelected(TergetPopID) = true;
                        minDistance = min(PopDecDistance(TergetPopID,:),minDistance);
                    end
                end
                Population = reducedPopulation;
            end
            Global.N = beforeN;
        end
    end
end

function Nich = decide_nich(PopDec,Global)
    N = size(PopDec,1);
    nb = min(3,max(N-1,0));
    
    % Automatically calculate niche radius
    % NormalizedPopDec = (PopDec - Global.lower)./(Global.upper - Global.lower);
    % d_dec = pdist2(NormalizedPopDec,NormalizedPopDec,'euclidean');
    d_dec = pdist2(PopDec,PopDec,'euclidean');
    temp = sort(d_dec);
    
    sigma_dec = sum(sum(temp(1:nb+1,:)))./(nb.*N);
    
    %{
    sigma_dec = reshape(temp(2:nb+1,:),size(temp(2:nb+1,:),1)*size(temp(2:nb+1,:),2),1);
    sigma_dec = median(sigma_dec);
    %}
    
    if sigma_dec == 0
        sigma_dec = inf;
    end
    
    neighbor_dec = d_dec < sigma_dec;
    Sh_dec = 1-d_dec./sigma_dec-eye(N);
    
    Nich = sum(Sh_dec.*neighbor_dec);
    Nich = transpose(Nich);
    
    %{
    sort_d_dec = sort(d_dec,2);
    median_d_dec = median(sort_d_dec(:,2:end),2);
    Nich = 1 ./ median_d_dec;
    %}
end

function OutputWeightedVectorsInfo(Global,W)
    OutputFolderPath = strcat(Global.outputFolderPath,'\otherOutputs');
    if ~isfolder(OutputFolderPath)
        mkdir(OutputFolderPath);
    end
    OutputMatrix(W, strcat(OutputFolderPath,'\WeightedVectors.dat'));
end

function  OutputSubPopulationInfo(Global,SubPopulation,SubPop_Niching_value,SubPop_Scalarizing_value)
    for i = 1 : size(SubPopulation, 1)
        % Output objective value and decition value of individual in subpopulation.
        OutputFolderPath = strcat(Global.outputFolderPath,'\otherOutputs\Subpopulation\WeightedVector',num2str(i));
        if ~isfolder(OutputFolderPath)
            mkdir(OutputFolderPath);
        end
        OutputMatrix(SubPopulation(i,:).objs, strcat(OutputFolderPath,'\',num2str(Global.evaluated),'Objs.dat'));
        OutputMatrix(SubPopulation(i,:).decs, strcat(OutputFolderPath,'\',num2str(Global.evaluated),'Vars.dat'));
        
        % Output Subpopulation space.
        OutputFolderPath = strcat(Global.outputFolderPath,'\otherOutputs\SubpopulationSpace\WeightedVector',num2str(i));
        if ~isfolder(OutputFolderPath)
            mkdir(OutputFolderPath);
        end
        OutputMatrix([SubPop_Niching_value(i,:);SubPop_Scalarizing_value(i,:)].', strcat(OutputFolderPath,'\',num2str(Global.evaluated),'Objs.dat'));
    end
end

