function ScalarizingFuncValue = MM2TScalarizingFunc(type,W,PopObjs,Z,Nadir)
    N = size(PopObjs,1);
    switch type
        case 1
            % PBI approach
            % theta = 5
            normW   = sqrt(sum(W.^2,2));
            normP   = sqrt(sum((PopObjs-repmat(Z,N,1)).^2,2));
            CosineP = sum((PopObjs-repmat(Z,N,1)).*W,2)./normW./normP;
            ScalarizingFuncValue   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
        case 2
            % Tchebycheff approach with normalization
            ScalarizingFuncValue = max(abs(PopObjs-repmat(Z,N,1))./repmat(Nadir - Z,N,1).*W,[],2);
        case 3
            % Tchebycheff approach
            ScalarizingFuncValue = max(abs(PopObjs-repmat(Z,N,1)).*W,[],2);
        case 4
            % Achievement Scalarizing Function with normalization
            ScalarizingFuncValue = max(abs(PopObjs-repmat(Z,N,1))./repmat(Nadir - Z,N,1)./W,[],2);
        case 5
            % Achievement Scalarizing Function
            ScalarizingFuncValue = max(abs(PopObjs-repmat(Z,N,1))./W,[],2);
        case 6
            % Weighted sum approach
            ScalarizingFuncValue = sum(abs(PopObjs-repmat(Z,N,1)).*W, 2);
    end
end

