import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { FiUserPlus, FiMenu, FiX } from 'react-icons/fi';
import styled, { keyframes } from 'styled-components';
import sciFiLogo from '../assets/sci-fi-logo.png';

// Keyframes for floating animation
const float = keyframes`
  0% { transform: translateY(0); }
  50% { transform: translateY(-20px); }
  100% { transform: translateY(0); }
`;

const Page = styled.div`
  height: 100vh;
  overflow: hidden;
  background: linear-gradient(135deg, #0d0d1a, #1a1a33);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const Particle = styled.div`
  position: absolute;
  border-radius: 50%;
  background: rgba(255,255,255,0.1);
  animation: ${float} ${({ duration }) => duration}s ease-in-out infinite;
`;

const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  position: relative;
  z-index: 1;
`;

const Card = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2.5rem;
  width: 320px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  color: #e0e0e0;
  text-align: center;
`;

const Logo = styled.img`
  width: 120px;
  margin-bottom: 1.5rem;
  animation: ${float} 6s ease-in-out infinite;
`;

const Title = styled.h2`
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  letter-spacing: 1px;
  text-transform: uppercase;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const Input = styled.input`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 0.95rem;
  outline: none;
  &::placeholder { color: rgba(255,255,255,0.7); }
`;

const Button = styled.button`
  margin-top: 1rem;
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #00d9a6, #00c096);
  color: #0d0d1a;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s ease;
  &:hover { background: linear-gradient(135deg, #00c096, #00d9a6); }
`;

const RegisterLink = styled.p`
  margin-top: 1.5rem;
  font-size: 0.9rem;
  & a { color: #00d9a6; text-decoration: none; font-weight: 500; }
  & a:hover { text-decoration: underline; }
`;

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [particles, setParticles] = useState([]);
  const navigate = useNavigate();

  const API_BASE_URL = 'http://192.168.0.150:8000';

  // Generate particles on mount
  useEffect(() => {
    const count = 50;
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push({
        id: i,
        size: Math.random() * 4 + 2,
        left: Math.random() * 100,
        top: Math.random() * 100,
        duration: Math.random() * 10 + 5,
        delay: Math.random() * 5
      });
    }
    setParticles(arr);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/users/token`,
        new URLSearchParams({ username, password }),
        { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
      );
      const { access_token, role } = response.data;
      if (!access_token || !role) throw new Error('Invalid response');
      localStorage.setItem('token', access_token);
      localStorage.setItem('role', role);
      switch (role) {
        case 'admin': navigate('/admin'); break;
        case 'security': navigate('/security'); break;
        case 'manager': navigate('/manager'); break;
        default: throw new Error('Invalid role');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
      localStorage.clear();
    } finally { setLoading(false); }
  };

  return (
    <Page>
      {particles.map(p => (
        <Particle
          key={p.id}
          style={{ width: p.size, height: p.size, left: `${p.left}%`, top: `${p.top}%`, animationDelay: `${p.delay}s` }}
          duration={p.duration}
        />
      ))}

      <Container>
        <Card>
          <Logo src={sciFiLogo} alt="Sci-Fi Logo" />
          <Title>SECURE PASS</Title>
          {error && <p style={{ color: '#ff6b6b' }}>{error}</p>}
          <Form onSubmit={handleSubmit}>
            <Input
              type="text"
              placeholder="Username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              disabled={loading}
              required
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              disabled={loading}
              required
            />
            <Button type="submit" disabled={loading}>
              {loading ? 'Logging in...' : 'Log In'}
            </Button>
          </Form>
          <RegisterLink>
            Need an account? <Link to="/register">Register</Link>
          </RegisterLink>
        </Card>
      </Container>
    </Page>
  );
};

export default Login;
