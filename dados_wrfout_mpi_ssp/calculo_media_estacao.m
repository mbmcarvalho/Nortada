function media_estacao = calculo_media_estacao(vel_media_diaria, estacao_indices)
    velocidade_estacao = vel_media_diaria(:, :, estacao_indices);
    media_estacao = mean(velocidade_estacao, 3);
end
