function season = month2season(month)
    seasons = {'Inverno', 'Primavera', 'Verão', 'Outono'};
    months_in_season = {[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]};
    
    % Encontrar a estação correspondente ao mês
    for i = 1:numel(months_in_season)
        if any(month == months_in_season{i})
            season = seasons{i};
            return;
        end
    end
    
    % Se o mês não estiver em nenhuma estação conhecida, retornar 'Desconhecido'
    season = 'Desconhecido';
end
