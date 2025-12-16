import httpx
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionError(Exception):
    """Excepción personalizada para errores de transacción"""
    pass

class Transaction:
    def __init__(self, email: str, password: str):
        if not email or not password:
            raise ValueError("Email y password son requeridos")
        
        if '@' not in email:
            raise ValueError("Email debe tener un formato válido")
        
        self.email = email
        self.password = password
        # Configurar que la API Key y Base URL se obtengan de un archivo de configuración o variables de entorno
        self.api_key = 'A9f$2kL!xQ7zP@eR6sVwT#1bGmN8uJcD'
        self.base_url = 'https://idptwinhibride.azurewebsites.net/api/v1/'
        self.token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self._auto_refresh_thread: Optional[threading.Thread] = None
        self._auto_refresh_stop_event = threading.Event()
        self._last_token_update: Optional[datetime] = None
        self._auto_refresh_enabled = False
        logger.info(f"Transaction instance created for user: {email}")

    def __del__(self):
        """Destructor para asegurar que el auto-refresh se detenga al eliminar la instancia"""
        try:
            if hasattr(self, '_auto_refresh_enabled') and self._auto_refresh_enabled:
                self.stop_auto_token_refresh()
        except Exception:
            pass  # Ignorar errores en el destructor

    def _make_request(self, method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]] = None, timeout: int = 30) -> httpx.Response:
        try:
            with httpx.Client(timeout=timeout) as client:
                if method.upper() == 'POST':
                    if 'Content-Type' in headers and headers['Content-Type'] == 'application/json':
                        if isinstance(payload, str):
                            response = client.post(url, data=payload, headers=headers)
                        else:
                            response = client.post(url, json=payload, headers=headers)
                    else:
                        response = client.post(url, data=payload, headers=headers)
                else:
                    response = client.get(url, headers=headers)
                
                return response
                
        except httpx.TimeoutException:
            logger.error(f"Timeout en la petición a {url}")
            raise TransactionError(f"Timeout en la petición. La operación tardó más de {timeout} segundos")
        except httpx.ConnectError:
            logger.error(f"Error de conexión a {url}")
            raise TransactionError("Error de conexión. Verifique su conexión a internet y la disponibilidad del servidor")
        except httpx.RequestError as e:
            logger.error(f"Error en la petición HTTP: {e}")
            raise TransactionError(f"Error en la petición HTTP: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado en la petición: {e}")
            raise TransactionError(f"Error inesperado: {str(e)}")

    def login(self) -> bool:
        """Realiza el login y obtiene los tokens de autenticación"""
        try:
            logger.info(f"Iniciando login para usuario: {self.email}")
            
            # Endpoint URL for login
            login_url = self.base_url + 'authentication/token'
            payload = {'email': self.email, 'password': self.password}
            headers = {'ApiHibridE': self.api_key, 'Content-Type': 'application/json'}

            response = self._make_request('POST', login_url, headers, payload)

            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'token' not in data or 'refreshToken' not in data:
                        logger.error("Respuesta del servidor no contiene los tokens esperados")
                        return False
                    
                    self.token = data.get('token')
                    self.refresh_token = data.get('refreshToken')
                    self._last_token_update = datetime.now()  # Registrar el tiempo de login
                    
                    if not self.token or not self.refresh_token:
                        logger.error("Tokens recibidos están vacíos")
                        return False
                    
                    logger.info('Login successful. Token obtained.')
                    logger.debug(f'Token: {self.token[:20]}...')  # Solo mostrar parte del token por seguridad
                    return True
                    
                except json.JSONDecodeError:
                    logger.error("Error al decodificar la respuesta JSON del servidor")
                    return False
                    
            elif response.status_code == 401:
                logger.error("Credenciales inválidas")
                print('Login failed: Credenciales inválidas')
                return False
            elif response.status_code == 403:
                logger.error("Acceso denegado")
                print('Login failed: Acceso denegado')
                return False
            elif response.status_code >= 500:
                logger.error(f"Error del servidor: {response.status_code}")
                print(f'Login failed: Error del servidor ({response.status_code})')
                return False
            else:
                logger.error(f"Login failed with status code: {response.status_code}")
                print(f'Login failed.')
                print(f'Status Code: {response.status_code}')
                print(f'Response Text: {response.text}')
                return False
                
        except TransactionError as e:
            logger.error(f"Error en login: {e}")
            print(f'Login failed: {e}')
            return False
        except Exception as e:
            logger.error(f"Error inesperado en login: {e}")
            print(f'Login failed: Error inesperado - {e}')
            return False

    def update_token(self) -> bool:
        """Actualiza el token de autenticación usando el refresh token"""
        try:
            if not self.refresh_token:
                logger.error('No refresh token available')
                print('No refresh token available. Please login first.')
                return False

            if not self.token:
                logger.error('No access token available')
                print('No access token available. Please login first.')
                return False

            logger.info("Actualizando token de autenticación")
            
            update_url = self.base_url + 'authentication/refresh-token'
            payload = json.dumps({'refreshToken': self.refresh_token})
            headers = {
                'ApiHibridE': self.api_key, 
                'Content-Type': 'application/json', 
                'Authorization': f'Bearer {self.token}'
            }

            response = self._make_request('POST', update_url, headers, payload)

            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'token' not in data or 'refreshToken' not in data:
                        logger.error("Respuesta del servidor no contiene los tokens esperados")
                        return False
                    
                    new_token = data.get('token')
                    new_refresh_token = data.get('refreshToken')
                    
                    if not new_token or not new_refresh_token:
                        logger.error("Nuevos tokens recibidos están vacíos")
                        return False
                    
                    self.token = new_token
                    self.refresh_token = new_refresh_token
                    self._last_token_update = datetime.now()  # Registrar el tiempo de actualización
                    logger.info('Token updated successfully.')
                    logger.debug(f'New Token: {self.token[:20]}...')  # Solo mostrar parte del token
                    return True
                    
                except json.JSONDecodeError:
                    logger.error("Error al decodificar la respuesta JSON del servidor")
                    return False
                    
            elif response.status_code == 401:
                logger.error("Token inválido o expirado")
                print('Token update failed: Token inválido o expirado. Realice login nuevamente.')
                return False
            elif response.status_code >= 500:
                logger.error(f"Error del servidor: {response.status_code}")
                print(f'Token update failed: Error del servidor ({response.status_code})')
                return False
            else:
                logger.error(f"Token update failed with status code: {response.status_code}")
                print('Token update failed.')
                print(f'Status Code: {response.status_code}')
                print(f'Response Text: {response.text}')
                return False
                
        except TransactionError as e:
            logger.error(f"Error en update_token: {e}")
            print(f'Token update failed: {e}')
            return False
        except Exception as e:
            logger.error(f"Error inesperado en update_token: {e}")
            print(f'Token update failed: Error inesperado - {e}')
            return False

    def start_auto_token_refresh(self, refresh_interval_hours: float = 10.0) -> bool:
        """
        Inicia el auto-refresh del token cada X horas
        
        Args:
            refresh_interval_hours: Intervalo en horas para refrescar el token (por defecto 10 horas)
            
        Returns:
            bool: True si se inició correctamente, False si ya estaba activo o hay error
        """
        if self._auto_refresh_enabled:
            logger.warning("Auto-refresh ya está activo")
            return False
            
        if not self.is_authenticated():
            logger.error("No se puede iniciar auto-refresh sin estar autenticado")
            return False
            
        self._auto_refresh_enabled = True
        self._auto_refresh_stop_event.clear()
        
        # Crear y iniciar el hilo de auto-refresh
        self._auto_refresh_thread = threading.Thread(
            target=self._auto_refresh_worker,
            args=(refresh_interval_hours,),
            daemon=True,
            name="TokenAutoRefresh"
        )
        self._auto_refresh_thread.start()
        
        logger.info(f"Auto-refresh de token iniciado. Intervalo: {refresh_interval_hours} horas")
        return True

    def stop_auto_token_refresh(self) -> bool:
        """
        Detiene el auto-refresh del token
        
        Returns:
            bool: True si se detuvo correctamente, False si no estaba activo
        """
        if not self._auto_refresh_enabled:
            logger.info("Auto-refresh no estaba activo")
            return False
            
        self._auto_refresh_enabled = False
        self._auto_refresh_stop_event.set()
        
        # Esperar a que termine el hilo (máximo 5 segundos)
        if self._auto_refresh_thread and self._auto_refresh_thread.is_alive():
            self._auto_refresh_thread.join(timeout=5.0)
            
        logger.info("Auto-refresh de token detenido")
        return True

    def _auto_refresh_worker(self, refresh_interval_hours: float):
        """
        Worker que ejecuta el auto-refresh en un hilo separado
        
        Args:
            refresh_interval_hours: Intervalo en horas para refrescar el token
        """
        refresh_interval_seconds = refresh_interval_hours * 3600  # Convertir a segundos
        
        logger.info(f"Worker de auto-refresh iniciado. Próxima actualización en {refresh_interval_hours} horas")
        
        while self._auto_refresh_enabled and not self._auto_refresh_stop_event.is_set():
            # Esperar el intervalo especificado o hasta que se solicite parar
            if self._auto_refresh_stop_event.wait(timeout=refresh_interval_seconds):
                # Se solicitó parar
                break
                
            # Verificar si aún estamos autenticados
            if not self.is_authenticated():
                logger.warning("Auto-refresh: Usuario no autenticado, deteniendo auto-refresh")
                self._auto_refresh_enabled = False
                break
                
            # Intentar actualizar el token
            try:
                logger.info("Auto-refresh: Actualizando token automáticamente")
                success = self.update_token()
                
                if success:
                    logger.info("Auto-refresh: Token actualizado exitosamente")
                else:
                    logger.error("Auto-refresh: Fallo al actualizar token")
                    # En caso de fallo, intentar nuevamente en la mitad del tiempo
                    if not self._auto_refresh_stop_event.wait(timeout=refresh_interval_seconds / 2):
                        continue
                    else:
                        break
                        
            except Exception as e:
                logger.error(f"Auto-refresh: Error inesperado al actualizar token: {e}")
                # En caso de error, esperar un tiempo menor antes del siguiente intento
                if not self._auto_refresh_stop_event.wait(timeout=refresh_interval_seconds / 4):
                    continue
                else:
                    break
        
        logger.info("Worker de auto-refresh terminado")

    def get_token_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el estado del token
        
        Returns:
            Dict con información del token: tiempo desde última actualización, auto-refresh activo, etc.
        """
        info = {
            'is_authenticated': self.is_authenticated(),
            'auto_refresh_enabled': self._auto_refresh_enabled,
            'last_update': None,
            'hours_since_update': None,
            'auto_refresh_thread_alive': False
        }
        
        if self._last_token_update:
            info['last_update'] = self._last_token_update.isoformat()
            hours_diff = (datetime.now() - self._last_token_update).total_seconds() / 3600
            info['hours_since_update'] = round(hours_diff, 2)
            
        if self._auto_refresh_thread:
            info['auto_refresh_thread_alive'] = self._auto_refresh_thread.is_alive()
            
        return info

    def logout(self) -> bool:
        """Realiza el logout, limpia los tokens y detiene el auto-refresh"""
        try:
            # Detener auto-refresh antes del logout
            if self._auto_refresh_enabled:
                logger.info("Deteniendo auto-refresh antes del logout")
                self.stop_auto_token_refresh()
            
            if not self.token:
                logger.error('No token available for logout')
                print('No token available. Please login first.')
                return False

            logger.info("Realizando logout")
            
            logout_url = self.base_url + 'authentication/logout'
            headers = {
                'ApiHibridE': self.api_key, 
                'Content-Type': 'application/json', 
                'Authorization': f'Bearer {self.token}'
            }
            payload = json.dumps({'refreshToken': self.refresh_token}) if self.refresh_token else json.dumps({})

            response = self._make_request('POST', logout_url, headers, payload)

            if response.status_code == 200:
                logger.info('Logout successful.')
                print('Logout successful.')
                self.token = None
                self.refresh_token = None
                self._last_token_update = None
                return True
            elif response.status_code == 401:
                logger.warning("Token ya expirado, limpiando tokens localmente")
                print('Logout: Token ya expirado, sesión limpiada.')
                self.token = None
                self.refresh_token = None
                self._last_token_update = None
                return True
            elif response.status_code >= 500:
                logger.error(f"Error del servidor durante logout: {response.status_code}")
                print(f'Logout failed: Error del servidor ({response.status_code})')
                # Aún así limpiar tokens localmente
                self.token = None
                self.refresh_token = None
                self._last_token_update = None
                return False
            else:
                logger.error(f"Logout failed with status code: {response.status_code}")
                print('Logout failed.')
                print(f'Status Code: {response.status_code}')
                print(f'Response Text: {response.text}')
                return False
                
        except TransactionError as e:
            logger.error(f"Error en logout: {e}")
            print(f'Logout failed: {e}')
            # En caso de error, aún así limpiar tokens localmente
            self.token = None
            self.refresh_token = None
            self._last_token_update = None
            return False
        except Exception as e:
            logger.error(f"Error inesperado en logout: {e}")
            print(f'Logout failed: Error inesperado - {e}')
            # En caso de error, aún así limpiar tokens localmente
            self.token = None
            self.refresh_token = None
            self._last_token_update = None
            return False

    def token_validation(self, partner_token: str) -> bool:
        if not self.token:
            logger.error('No token available for validation')
            return False

        try:
            logger.info("Validando token de autenticación")
            
            validate_url = self.base_url + 'authentication/token-validation'
            headers = {
                'ApiHibridE': self.api_key, 
                'Content-Type': 'application/json', 
                'Authorization': f'Bearer {self.token}'
            }
            payload = json.dumps({'token': partner_token})

            response = self._make_request('POST', validate_url, headers, payload)

            if response.status_code == 200:
                data = response.json()
                logger.info(data.get('message'))
                return True
            elif response.status_code >= 400 and response.status_code < 500:
                logger.warning("Token inválido o expirado")
                return False
            elif response.status_code >= 500:
                logger.error(f"Error del servidor durante validación de token: {response.status_code}")
                return False
            else:
                logger.error(f"Token validation failed with status code: {response.status_code}")
                return False
                
        except TransactionError as e:
            logger.error(f"Error en token_validation: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado en token_validation: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Verifica si el usuario está autenticado"""
        return self.token is not None and self.refresh_token is not None

    def get_auth_headers(self) -> Optional[Dict[str, str]]:
        """Obtiene los headers de autenticación para peticiones autenticadas"""
        if not self.is_authenticated():
            return None
        
        return {
            'ApiHibridE': self.api_key,
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }