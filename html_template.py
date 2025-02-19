css = '''
<style> 
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #f8f9fa;
}
.chat-message.user {
    background-color: #fffde7;  /* Amarillo claro de fondo, legible */
    border-left: 5px solid #ffe57f;  /* Borde amarillo claro */
}
.chat-message.bot {
    background-color: #ffebee;  /* Fondo rojo claro */
    border-left: 5px solid #ef5350;  /* Borde rojo más oscuro */
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 68px;
    max-height: 68px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #212529;
}
'''

logo = '''
<div style="margin-bottom: 15px; text-align: center;">
    <img src="https://seeklogo.com/images/A/alcaldia-mayor-de-bogota-logo-CA468F866B-seeklogo.com.png" 
         alt="Logo Alcaldia BOgota" style="max-width: 18%; height: auto;">
    <img src="https://i.ibb.co/cgQbQTQ/logo.png" 
        alt="Logo GOVLAB" style="max-width: 20%; height: auto;">
</div>
'''
