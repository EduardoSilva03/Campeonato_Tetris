import cv2
import numpy as np
import os

DEBUG_MODE = False

def detectar_peca(roi_frame, pecas_info, sensibilidade):
    """
    Função reutilizável para detectar qual peça está em uma dada ROI.
    Retorna o nome da peça ou None.
    """
    if roi_frame is None or roi_frame.size == 0:
        return None
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    for nome_peca, info in pecas_info.items():
        mask = None
        if nome_peca == 'Z':
            lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            mask = mask1 + mask2
        else:
            lower_bound, upper_bound = info[1], info[2]
            mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

        if cv2.countNonZero(mask) > sensibilidade:
            return nome_peca
    return None

def main():
    # ROI Padrão
    #roi_next = (80, 140, 260, 410)
    #roi_hold = (80, 140, 40, 160)
    # (Y inicial, Y final, X inicial, X final)
    # (borda de CIMA, borda de BAIXO, borda da ESQUERDA, borda da DIREITA)

    roi_next = (80, 170, 240, 410)
    roi_hold = (80, 170, 20, 160)

    #Cores
    pecas_info = {
        'I': ['red-bull-tetris-i-tetrimino.jpg', np.array([90, 80, 80]), np.array([110, 255, 255])],
        'O': ['red-bull-tetris-o-tetrimino.jpg', np.array([25, 120, 120]), np.array([35, 255, 255])],
        'T': ['red-bull-tetris-t-tetrimino.jpg', np.array([140, 80, 80]), np.array([160, 255, 255])],
        'J': ['red-bull-tetris-j-tetrimino.jpg', np.array([110, 120, 80]), np.array([130, 255, 255])],
        'L': ['red-bull-tetris-l-tetrimino.jpg', np.array([10, 120, 120]), np.array([20, 255, 255])],
        'S': ['red-bull-tetris-s-tetrimino.jpg', np.array([50, 80, 80]), np.array([70, 255, 255])],
        'Z': ['red-bull-tetris-z-tetrimino.jpg'],
    }

    video_path = 'tetris.mp4'
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video = cv2.VideoWriter('tetris_analisado_finalizado.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Variáveis contagem
    pecas_jogadas_count = 0
    peca_anterior_next = None
    peca_anterior_hold = None
    historico_pecas = []

    print("Iniciando análise com lógica final...")

    # Pula os primeiros frames para ignorar a tela de loading
    for _ in range(int(fps * 2)):
        ret, frame = cap.read()
        if not ret: break

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        roi_next_frame = frame[roi_next[0]:roi_next[1], roi_next[2]:roi_next[3]]
        roi_hold_frame = frame[roi_hold[0]:roi_hold[1], roi_hold[2]:roi_hold[3]]

        peca_atual_next = detectar_peca(roi_next_frame, pecas_info, 600)
        peca_atual_hold = detectar_peca(roi_hold_frame, pecas_info, 600)

        # Lógica peça repetida na contagem
        
        peca_entrou_no_hold = peca_atual_hold is not None and peca_atual_hold != peca_anterior_hold
        peca_saiu_do_hold = peca_atual_hold is None and peca_anterior_hold is not None

        peca_saiu_do_next = (peca_atual_next is not None and 
                             peca_anterior_next is not None and 
                             peca_atual_next != peca_anterior_next)

        if peca_entrou_no_hold:
            if pecas_jogadas_count > 0:
                pecas_jogadas_count -= 1
                if historico_pecas: historico_pecas.pop()

        if peca_saiu_do_hold:
            pecas_jogadas_count += 1
            historico_pecas.append(peca_anterior_hold)
        elif peca_saiu_do_next and not peca_entrou_no_hold:
            pecas_jogadas_count += 1
            historico_pecas.append(peca_anterior_next)

        peca_anterior_next = peca_atual_next
        peca_anterior_hold = peca_atual_hold

        # Visualizar o ROI
        cv2.rectangle(frame, (roi_next[2], roi_next[0]), (roi_next[3], roi_next[1]), (0, 255, 0), 2)
        cv2.rectangle(frame, (roi_hold[2], roi_hold[0]), (roi_hold[3], roi_hold[1]), (0, 255, 255), 2)
        
        texto_contador = f"Pecas Jogadas: {pecas_jogadas_count}"
        texto_next = f"NEXT: {peca_atual_next if peca_atual_next else 'Nenhum'}"
        texto_hold = f"HOLD: {peca_atual_hold if peca_atual_hold else 'Nenhum'}"
        
        cv2.putText(frame, texto_contador, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, texto_contador, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, texto_next, (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, texto_next, (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, texto_hold, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, texto_hold, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not DEBUG_MODE:
            output_video.write(frame)
        
        if DEBUG_MODE:
            cv2.imshow('Video Principal', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
    print("\nAnálise concluída!")
    print(f"Total de peças jogadas: {pecas_jogadas_count}")
    print(f"Sequência de peças jogadas: {' -> '.join(str(p) for p in historico_pecas if p is not None)}")

if __name__ == '__main__':
    main()