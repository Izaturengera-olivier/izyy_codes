import { useState } from "react";
import axios from "axios";
import {Link} from "react-router-dom";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");

  const handleRequestReset = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:8000/password-reset/", { email });
      setMessage(response.data.message);
    } catch (error) {
      setMessage(error.response?.data?.error || "Something went wrong");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
        <form onSubmit={handleRequestReset} className="bg-white p-6 shadow-lg rounded">
            <h2 className="text-2xl font-bold">Forgot Password?</h2>
            <p>Enter your email to receive a verification code.</p>
            <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-2 border rounded mt-2"
                required
            />
            <button type="submit" className="w-full bg-blue-500 text-white py-2 rounded mt-3">
                Send Code
            </button>
            {message && <p className="text-green-500 mt-3">{message}</p>}
            <p>
                Did you receive a code? <Link to="/reset-password/" className="text-blue-500 hover:underline">Reset Password</Link>
            </p>
        </form>
    </div>
  );
};

export default ForgotPassword;
