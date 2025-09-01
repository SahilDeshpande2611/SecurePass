import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import styled, { keyframes } from 'styled-components';
import sciFiLogo from '../assets/sci-fi-logo.png';

// Floating animation keyframes
const float = keyframes`
  0% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0); }
`;

// Styled Components
const PageWrapper = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #0d0d1a, #1a1a33);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const Card = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2.5rem;
  width: 360px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  color: #e0e0e0;
  text-align: center;
`;

const Logo = styled.img`
  width: 100px;
  margin-bottom: 1.5rem;
  animation: ${float} 6s ease-in-out infinite;
`;

const Title = styled.h2`
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1rem;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: #00d9a6;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
`;

const Input = styled.input`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;
  &::placeholder { color: rgba(255,255,255,0.7); }
`;

const Select = styled.select`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;
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

const ErrorText = styled.p`
  color: #ff6b6b;
  margin-top: -0.5rem;
`;

const RedirectText = styled.p`
  margin-top: 1rem;
  font-size: 0.9rem;
  color: #a0a8b7;
  & > a { color: #00d9a6; text-decoration: none; font-weight: 500; }
  & > a:hover { text-decoration: underline; }
`;

const Register = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('security');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const API_BASE_URL = 'http://192.168.0.150:8000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!username || !password) {
      setError('Username and password are required');
      return;
    }
    try {
      await axios.post(
        `${API_BASE_URL}/users/register`,
        { username, password, role }
      );
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
    }
  };

  return (
    <PageWrapper>
      <Card>
        <Logo src={sciFiLogo} alt="Sci-Fi Logo" />
        <Title>Register</Title>
        {error && <ErrorText>{error}</ErrorText>}
        <Form onSubmit={handleSubmit}>
          <Input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <Select
            value={role}
            onChange={(e) => setRole(e.target.value)}
          >
            <option value="admin">Admin</option>
            <option value="security">Security</option>
            <option value="manager">Manager</option>
          </Select>
          <Button type="submit">Register</Button>
        </Form>
        <RedirectText>
          Already have an account? <Link to="/login">Login</Link>
        </RedirectText>
      </Card>
    </PageWrapper>
  );
};

export default Register;
